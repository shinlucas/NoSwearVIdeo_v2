from flask import render_template, Flask, request, redirect, url_for
import threading
import initializer  # 새로 만든 초기화 모듈
import replace_swear
import download_live
import download_video

app = Flask(__name__)

@app.before_first_request
def before_first_request():
    initializer.initialize_once()

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        video_link = request.form['video_link']
        print(f"Received video link in form: {video_link}")  # URL 로그 출력
        return redirect(url_for('load', video_link=video_link))
    return render_template('index.html')

@app.route('/load')
def load():
    video_link = request.args.get('video_link')
    print(f"Received video link in load: {video_link}")  # URL 로그 출력
    if video_link is None:
        return "Error: No video link provided", 400
    process_video(video_link)
    return render_template('load.html')

@app.route('/video')
def video():
    return render_template('video.html')

def process_video(video_link):
    print(f"Processing video link: {video_link}")
    download_thread = threading.Thread(target=download_and_start_replace, args=(video_link,))
    download_thread.start()

# 실시간 환경은 수정 필요
def download_and_start_replace(video_link):
    print(f"Starting download for: {video_link}")
    #download_video.download_chunk(video_link)
    #target_dict = initializer.get_target_dict()
    #replace_thread = threading.Thread(target=replace_swear.main_part, args=(target_dict,))
    #replace_thread.daemon = True  # 메인 스레드가 종료되더라도 이 스레드는 계속 실행됨
    #replace_thread.start()
    target_dict = initializer.get_target_dict()

    #stream_thread = threading.Thread(target=run_stream_thread, args=(video_link,))
    replace_thread = threading.Thread(target=run_replace_thread, args=(target_dict,))
    
    #stream_thread.daemon = True 
    replace_thread.daemon = True  # 메인 스레드가 종료되더라도 이 스레드는 계속 실행됨
    
    #stream_thread.start()
    print("Stream thread started.")
    replace_thread.start()
    print("Replace thread started.")

    #stream_thread.join()  # 스레드가 끝날 때까지 대기
    replace_thread.join()

def run_stream_thread(video_link):
    print("Stream thread is running.")
    download_live.capture_stream(video_link)
    print("Stream thread completed.")

def run_replace_thread(target_dict):
    print("Replace thread is running.")
    replace_swear.main_part(target_dict)
    print("Replace thread completed.")

if __name__ == '__main__':
    initializer.initialize_once()
    app.run(host='127.0.0.1', port=8080, debug=False, use_reloader=False)

# Insert 'waitress-serve --port=8080 app:app' to Terminal
# Enter this link : http://127.0.0.1:8080
# https://www.youtube.com/watch?v=2SwvjXaTzEM
