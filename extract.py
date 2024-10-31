import os
import subprocess
import argparse

class InvalidPathError(Exception):
    pass

def extract_frames(video_path, output_path, video_name):
    max_frames = 5
    command = [
        'ffmpeg', '-i', video_path, '-vf', 'fps=1/3',
        f'{output_path}/{video_name}_frame_%04d.jpg'
    ]

    try:
        subprocess.run(command, text=True, capture_output=True, check=True)
        print(f"[완료] {video_path}에서 이미지 추출 완료.")

        extracted_files = [
            f for f in os.listdir(output_path)
            if f.startswith(video_name) and f.endswith('.jpg')
        ]
        # print(len(extracted_files))
        if len(extracted_files) > max_frames:
            for file in extracted_files[max_frames:]:
                os.remove(os.path.join(output_path, file))

    except subprocess.CalledProcessError as e:
        print(f"[오류] {video_path} 처리 중 문제 발생: {e.stderr}")

def process_all_videos(args):
    input_folder = args.input_dir
    output_folder = args.output_dir

    # 결과 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(input_folder):
        raise InvalidPathError(f"'{input_folder}' : 디렉토리 경로가 올바르지 않습니다.")
    for video_file in os.listdir(input_folder):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(input_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            extract_frames(video_path, output_folder, video_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/input",
                        help="이미지를 추출할 동영상이 저장된 디렉토리 경로 (기본값: ./data/input)")
    parser.add_argument("--output_dir", type=str, default="./data/output",
                        help="추출된 이미지가 저장된 디렉토리 경로 (기본값: ./data/output)")
    args = parser.parse_args()
    process_all_videos(args)

