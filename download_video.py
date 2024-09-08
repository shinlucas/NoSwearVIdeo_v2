import os
import subprocess
from yt_dlp import YoutubeDL

def download_video(url, output_dir):
    print(f"Downloading video from: {url}")  # URL 로그 출력
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(output_dir, 'video.%(ext)s'),
        'extractor-args': 'youtube:player_client=android'
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def split_video(input_file, output_dir, chunk_size=20):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = float(result.stdout)
    
    if duration <= 10:
        print("Video duration is 10 seconds or less. No chunks will be created.")
        return
    
    print('duration : ', duration)

    num_chunks = int(duration // chunk_size)
    print('num_chunks :', num_chunks)
    for i in range(num_chunks):
        start_time = i * chunk_size
        output_file = os.path.join(output_dir, f'video_{i:01d}.mp4')
        print(f"Creating chunk: {output_file} from {start_time} to {start_time + chunk_size}")
        subprocess.run(['ffmpeg', '-i', input_file, '-ss', str(start_time), '-t', str(chunk_size), '-c:v', 'libx264', '-c:a', 'aac', output_file])
    
    # Handle the remaining part of the video
    if duration % chunk_size != 0:
        start_time = num_chunks * chunk_size
        output_file = os.path.join(output_dir, f'video_{num_chunks:01d}.mp4')
        remaining_time = duration - start_time
        if remaining_time >= 10:
            print(f"Creating remaining chunk: {output_file} from {start_time} to end ({remaining_time} seconds)")
            subprocess.run(['ffmpeg', '-i', input_file, '-ss', str(start_time), '-t', str(remaining_time), '-c:v', 'libx264', '-c:a', 'aac', output_file])


def download_chunk(url):
    print(f"download_chunk called with URL: {url}")  # URL 로그 출력
    download_video(url, "./video")
    input_file = "./video/video.mp4"
    split_video(input_file, "./video")