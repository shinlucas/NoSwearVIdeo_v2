import os
import time
from yt_dlp import YoutubeDL

def capture_stream(url, chunk_duration=20, output_dir='./video'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'no_warnings': True,
        'outtmpl': '-',
        'force_generic_extractor': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_url = info['url']

    # ffmpeg 명령어를 사용하여 스트림을 연속적으로 캡처
    command = f'ffmpeg -i "{video_url}" -c copy -f segment -segment_time {chunk_duration} -reset_timestamps 1 "{output_dir}/video_%01d.mp4"'
    os.system(command)

