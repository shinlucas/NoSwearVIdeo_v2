import initial
import os
import time
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch
from moviepy.editor import VideoFileClip, AudioFileClip
import soundfile as sf
import subprocess
import io
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import numpy as np

Video_Flag = 0
n = 0

# 실행될 부분
def main_part(word_list):
    global Video_Flag
    global n
    global video_file
    global audio_file

    time.sleep(60)

    while True:

        if not os.path.isfile(initial.video_path + f"/video_{n}.mp4") and Video_Flag <= 200:  # 파일이 존재하지 않는 경우
            time.sleep(1)
            Video_Flag += 1
            continue
        elif os.path.isfile(initial.video_path + f"/video_{n}.mp4") and Video_Flag <= 200:  # 파일이 존재하는 경우
            video_file = initial.video_path + f"/video_{n}.mp4"
            video_to_audio(video_file)  # 오디오 파일 생성
            audio_file = initial.audio_path + f"/audio_{n}.wav"
            Video_Flag = 0
            video_conversion(word_list)
            #clear_temp_directory()
            n += 1
        elif Video_Flag > 200:
            print("Flag Over")
            break

# Audio file 마다 수행
def video_conversion(word_list):
    transcription = audio_to_text(audio_file)  # transcription 생성
    target_dict_sorted = sorted(word_list, key=len, reverse=True) # 큰 단어를 먼저 수행하기 위함

    # 수정1
    global main_sentence
    main_sentence = "".join(segment["text"] for segment in transcription["segments"])
    print("main_sentence:", main_sentence)
    voice_cloning = True
    global replace_segment

    #f = open("./text.txt", 'a')

    for segment in transcription['segments']:
        text = segment['text']
        start_time = segment['start']
        end_time = segment['end']
        replace_segment = text
        print("text:", text)
        #f.write("text: ")
        #f.write(text)
        #f.write("\n")
        voice_index = False
        
        for target_word in target_dict_sorted:
            index = replace_segment.find(target_word)
            while index != -1:
                if voice_cloning:
                    # 원 음성파일을 학습
                    reference_speaker = f'./audio/audio_{n}.wav'  # 복제하려는 음성 파일 경로
                    target_se, audio_name = se_extractor.get_se(reference_speaker, initial.tone_color_converter, vad=False)
                    voice_cloning = False
                    print("voice_cloning when audio:", n, "text:" ,text)
                replace_word(target_word, target_dict_sorted) # 텍스트만 보내면 알아서 대체어 찾아서 수정해오는 새로운 텍스트 생성
                print("replace_segment:", replace_segment)
                #f.write("replace_segment: ")
                #f.write(replace_segment)
                #f.write("\n")

                voice_index = True
                index = text.find(target_word, index + 1)
        
        # 해당 텍스트에 음성 적용
        if voice_index:
            voice_model = initial.tts_model
            speaker_ids = voice_model.hps.data.spk2id
            voice_output_dir = "./temp"
            device = "cpu"
            src_path = voice_output_dir + f"/temp.wav"

            for speaker_key in speaker_ids.keys():
                speaker_id = speaker_ids[speaker_key]
                speaker_key = speaker_key.lower().replace('_', '-')
        
                source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
                voice_model.tts_to_file(replace_segment, speaker_id, src_path, speed=1.2)
                save_path = f'{voice_output_dir}/audio_token.wav'

                # Run the tone color converter
                encode_message = "@MyShell"
                initial.tone_color_converter.convert(
                    audio_src_path=src_path, 
                    src_se=source_se, 
                    tgt_se=target_se, 
                    output_path=save_path,
                    message=encode_message)
            
            # temp 음성 생성해서 적용()
            if not os.path.exists(initial.audio_path + f"/final_audio_{n}.wav"):
                volume_equal(start_time, end_time, audio_file, initial.temp_path + "/audio_token.wav", initial.audio_path + f"/final_audio_{n}.wav")
            else:
                volume_equal(start_time, end_time, initial.audio_path + f"/final_audio_{n}.wav", initial.temp_path + "/audio_token.wav", initial.audio_path + f"/final_audio_{n}.wav")
    
    # 최종 음성을 영상에 합성
    if not os.path.exists(initial.audio_path + f"/final_audio_{n}.wav"):
        shutil.copy(video_file, initial.output_path + f"/final_video_{n}.mp4")
    else:
        replace_video(video_file, initial.audio_path + f"/final_audio_{n}.wav", initial.output_path + f"/final_video_{n}.mp4")

# Audio 추출
def video_to_audio(video_path):
    audio_path = initial.audio_path + f"/audio_{n}.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        audio_path
    ]
    subprocess.run(command, check=True)

def replace_word(target_word, target_dict_sorted):
    global main_sentence
    global replace_segment

    index = main_sentence.find(target_word)

    sentence1 = main_sentence[:index]
    sentence2 = main_sentence[index+len(target_word):]
    candidates = predict_next_word(sentence1, sentence2, target_dict_sorted)
    best_cand = select_best_candidate(candidates, sentence1, sentence2)
    replace_segment = replace_segment.replace(target_word, best_cand, 1)
    main_sentence = main_sentence.replace(target_word, best_cand, 1)
    print("candidates:", candidates)
    print("best:", best_cand)


# transcription 생성
def audio_to_text(audio_path):
    result = initial.wisper_model.transcribe(audio_path, verbose=True, language='ko')
    return result


# 대체어 예측 함수
def is_hangul_syllable(word):
    # 한글 자음이나 모음만 있는 경우를 제외하기 위한 함수
    for char in word:
        if not ('가' <= char <= '힣'):
            return False
    return True

def predict_next_word(first_part, second_part, word_list):
    text = f"{first_part}[MASK]{second_part}"
    print(text)
    inputs = initial.replace_tokenizer(text, return_tensors='pt')
    mask_index = torch.where(inputs["input_ids"] == initial.replace_tokenizer.mask_token_id)[1].item()

    with torch.no_grad():
        outputs = initial.replace_model(**inputs)
        predictions = outputs.logits[0, mask_index].topk(70)
        predicted_token_ids = predictions.indices.tolist()

    predicted_tokens = initial.replace_tokenizer.convert_ids_to_tokens(predicted_token_ids)
    not_word = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '=', '+', '[', ']', '{', '}', ';', ':', '\'', '\"', ',', '.', '<', '>', '/', '?', '\\', '|', '`', '~']
    pred = []
    ind = 0

    while len(pred) != 5 and ind != len(predicted_tokens):
        ox = 0
        word = predicted_tokens[ind]
        for i in not_word:
            if i in word:
                ox = 1
                break
        # 한글 자음이나 모음만 있는 경우를 제외하고, 한 글자 단어와 word_list에 있는 단어를 제외
        if ox == 0 and len(word) > 1 and is_hangul_syllable(word): # and word not in word_list
            pred.append(word)
        ind += 1

    return pred

def calculate_token_probability(sentence, candidate, mask_token_index):
    inputs = initial.choice_tokenizer(sentence, return_tensors='pt')
    mask_token_logits = initial.choice_model(**inputs).logits
    mask_token_logits = mask_token_logits[0, mask_token_index, :]
    
    # 후보 단어의 토큰 ID를 구합니다.
    candidate_token_id = initial.choice_tokenizer.convert_tokens_to_ids(candidate)
    
    # 후보 단어의 확률을 계산합니다.
    candidate_token_logit = mask_token_logits[candidate_token_id].item()
    return candidate_token_logit

def select_best_candidate(candidates, sent_1, sent_2):
    best_candidate = None
    best_score = float('inf')  # 높은 확률(로그 확률)이 더 좋은 선택이 됩니다.
    
    for candidate in candidates:
        candidate_sentence = sent_1 + candidate + sent_2
        # [MASK] 토큰이 있는 위치를 찾습니다.
        mask_token_index = len(initial.choice_tokenizer.tokenize(sent_1))  # [MASK] 토큰이 있는 위치
        score = calculate_token_probability(candidate_sentence, candidate, mask_token_index)
        print(candidate, " score :", score)
        if score < best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


# 기존 음성에 새로운 음성 합성
def replace_audio_segment(input_audio_path, replacement_audio_path, timestamp, output_path):
    original_audio = AudioSegment.from_wav(input_audio_path)
    replacement_audio = AudioSegment.from_wav(replacement_audio_path)

    start_ms = timestamp[0] * 1000
    end_ms = timestamp[1] * 1000
    original_audio = original_audio[:start_ms] + AudioSegment.silent(duration=(end_ms - start_ms)) + original_audio[end_ms:]
    overlay_audio = original_audio.overlay(replacement_audio, position=start_ms)
    original_audio = overlay_audio
    
    original_audio.export(output_path, format="wav")
    
# video 교체
def replace_video(video_path, new_audio_path, output_path):
    try:
        video = VideoFileClip(video_path)
        new_audio = AudioFileClip(new_audio_path)
        video_with_new_audio = video.set_audio(new_audio)
        
        # 비디오를 저장할 때의 파라미터 설정
        video_with_new_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Video with new audio saved to {output_path}")
    except Exception as e:
        print(f"Error occurred while replacing video audio: {e}")


# temp_path 비우기
def clear_temp_directory():
    temp_path = initial.temp_path
    
    # 디렉토리 내 모든 파일 제거
    for filename in os.listdir(temp_path):
        file_path = initial.temp_path + "/" + filename
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 파일 또는 심볼릭 링크 제거
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 디렉토리 제거
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def match_audio_volume(source_segment, target_segment):
    # 타겟 음성의 음량을 소스 음성의 음량에 맞춤
    source_rms = source_segment.rms
    target_rms = target_segment.rms
    
    if target_rms == 0:
        return target_segment  # 타겟 음성이 무음인 경우
    # 음량 차이에 따라 dB 조절
    change_in_db = 20 * np.log10(source_rms / target_rms)
    return target_segment + change_in_db

def stretch_audio_to_duration(audio_segment, target_duration_ms):
    # 타겟 길이에 맞게 속도 변경
    duration_ratio = target_duration_ms / len(audio_segment)
    
    if duration_ratio < 0.5 or duration_ratio > 2.0:
        print(f"Warning: duration_ratio of {duration_ratio} might degrade audio quality.")
    
    if duration_ratio > 1:
        # 속도를 줄여야 할 경우 (길이를 늘려야 하는 경우)
        new_sample_rate = int(audio_segment.frame_rate / duration_ratio)
        stretched_audio = audio_segment._spawn(audio_segment.raw_data, overrides={'frame_rate': new_sample_rate})
        stretched_audio = stretched_audio.set_frame_rate(audio_segment.frame_rate)
    else:
        # 속도를 높여야 할 경우 (길이를 줄여야 하는 경우)
        stretched_audio = audio_segment.speedup(playback_speed=1/duration_ratio, chunk_size=150, crossfade=25)
    
    return stretched_audio

def volume_equal(start, end, source_file, temp_file, return_file):
    source_audio = AudioSegment.from_wav(source_file)
    target_audio = AudioSegment.from_wav(temp_file)
    # 특정 타임스탬프 구간에서 음량 계산 (예: 10초에서 15초 사이)
    start_time = start * 1000  # 밀리초 단위
    end_time = end * 1000
    target_duration = end_time - start_time
    print("duration:", target_duration)
    print("temp_len:", len(target_audio))

    stretched_target_audio = stretch_audio_to_duration(target_audio, target_duration)

    print("after_trans:", len(stretched_target_audio))

    # 타겟 음성의 음량을 소스 음성의 특정 구간 음량에 맞춤  
    adjusted_target_audio = match_audio_volume(source_audio[start_time:end_time], stretched_target_audio)

    # 타겟 음성을 원본 음성 파일에 삽입
    output_audio = source_audio[:start_time] + adjusted_target_audio + source_audio[end_time:]

    # 결과를 파일로 저장
    output_audio.export(return_file, format="wav")