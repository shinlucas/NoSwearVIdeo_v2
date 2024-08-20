import os
import time
from yt_dlp import YoutubeDL

def capture_stream(url, chunk_duration=20, output_dir='./video'):
    """
    yt-dlp와 ffmpeg를 사용하여 실시간 스트리밍을 20초마다 저장하는 함수
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'no_warnings': True,
        'outtmpl': '-',
        'force_generic_extractor': True
    }

    ind = 0
    while True:
        output_file = output_dir + f'/video_{ind}.mp4'
        print(f"Capturing stream to: {output_file} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info['url']

        # ffmpeg 명령어를 사용하여 스트림을 20초씩 캡처
        command = f'ffmpeg -i "{video_url}" -t {chunk_duration} -c copy "{output_file}"'
        os.system(command)

        ind += 1
        time.sleep(chunk_duration)
