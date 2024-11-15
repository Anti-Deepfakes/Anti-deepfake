from pathlib import Path
import httpx
# from app.models.feedback import Feedback
from fastapi import UploadFile
from io import BytesIO
import numpy as np
import cv2
import os
from insightface.app import FaceAnalysis
from app.models.preprocessing import PreprocessingEntity
from sqlalchemy.orm import Session
import shutil


class PhotoService:
    @staticmethod
    def file_upload(file: UploadFile) -> dict:
        result = {
            "deepfakeProbability": -1,
            "feedbackId": None
        }

        if not file.filename:
            return result

        try:
            # Deepfake 확률 계산
            deepfake_probability = PhotoService.photo_detection(file)


            return result
        except Exception as e:
            print(f"Error during file upload: {e}")
            return result

    @staticmethod
    def photo_detection(file: UploadFile) -> float:
        
        url = "https://anti-deepfake.kr/detect/photo/detection"
        try:
            with httpx.Client() as client:
                file_type = "image/jpeg" if file.filename.endswith(".jpg") else "image/png"
                
                # 'file'에 대해 올바른 MIME 타입을 지정하여 전송
                files = {"file": (file.filename, file.file, file_type)}
                response = client.post(url, files=files)
                response.raise_for_status()

                # 응답에서 결과 추출
                data = response.json()
                result = data.get("result", -1)
                print(f"Detection result: {result}")
                return result
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
        except Exception as e:
            print(f"Error during photo detection: {e}")
        return -1  # 오류 발생 시 -1 반환
    @staticmethod
    def preprocessing(image, db: Session):
        # 이미지를 메모리에서 읽기
        image_bytes = image.file.read()

        # 얼굴 분석 객체 준비
        face_detector = FaceAnalysis(name='buffalo_l')
        face_detector.prepare(ctx_id=1, det_size=(224, 224))

        # 이미지를 numpy 배열로 변환
        np_img = np.frombuffer(image_bytes, np.uint8)
        image_cv = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # 이미지를 RGB로 변환하고 리사이즈
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (224, 224))

        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = np.transpose(image_normalized, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        # 얼굴 탐지
        faces = face_detector.get(image_cv)
        
        try:
            face = faces[0]
            bbox = face['bbox']
            landmarks = face['landmark_3d_68']
        except Exception as e:
            print(e, " : Error in image processing")
            bbox = [0, 0, 0, 0]
            landmarks = []

        weight_map = np.zeros(image_normalized.shape[1:], dtype=np.float32)
        weight_map = add_weight_to_bbox(weight_map, bbox)
        weight_map = add_weight_to_landmarks(weight_map, landmarks)
        print("end preprocessing")
        # 이미지와 .npz 파일을 저장할 디렉토리 경로
        directory = '/home/ubuntu/data/disrupt/tmp/'
        print("start saving")
        # 디렉토리에 있는 파일 목록을 확인하여 마지막 번호를 추적
        existing_files = os.listdir(directory)
        existing_files = [f for f in existing_files if f.endswith('.npz')]  # .npz 파일만 필터링
        existing_indexes = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
        next_index = max(existing_indexes, default=-1) + 1  # 다음 번호

        # 파일명 설정 (000.npz 형식)
        file_name = f"{next_index:03d}"

        # .npz 파일 저장
        np.savez(os.path.join(directory, f"{file_name}.npz"), image_normalized=image_normalized, weight_map=weight_map, 
                 bbox=np.array(bbox, dtype=np.float32), landmarks=np.array(landmarks, dtype=np.float32) if len(landmarks) > 0 else np.zeros((68, 3), dtype=np.float32))


        print(f"Saved {file_name}.npz and {file_name}.jpg")
        print("end saving")
        print("start Database Save")
        npz_url = os.path.join(directory, f"{file_name}.npz")
        create_preprocessing(db, npz_url,True,None)

        if int(file_name[-1:])>=9:
            print("do Trigger")
            do_trigger(db)
            pass
            
        
        # 반환값으로 파일명 또는 성공 메시지를 반환
        return {"message": "Preprocessing successful", "filename": f"{file_name}.npz"}
# 디렉토리 버전 관리 함수
def get_next_version(directory: str):
    """디렉토리에서 다음 버전 번호를 찾는 함수"""
    version_files = [f for f in os.listdir(directory) if f.startswith("ver")]
    if not version_files:
        return "ver000"  # 첫 번째 버전
    # 가장 큰 버전 번호를 찾아서 1 증가
    version_numbers = [int(f[3:]) for f in version_files]
    next_version = max(version_numbers) + 1
    return f"ver{next_version:03d}"  # ver001, ver002, ...

def move_file(source_path: str, target_dir: str, ver_name: str):
    """파일을 target 디렉토리의 지정된 버전 디렉토리로 이동"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # 디렉토리 없으면 생성
    
    # 타겟 디렉토리 안에 해당 버전 폴더 생성
    target_version_dir = os.path.join(target_dir, ver_name)
    if not os.path.exists(target_version_dir):
        os.makedirs(target_version_dir)
    
    # 파일을 버전 디렉토리로 이동
    target_path = os.path.join(target_version_dir, os.path.basename(source_path))
    shutil.move(source_path, target_path)
    
    # 이동 후 새로운 경로를 반환
    return target_path

def do_trigger(db: Session):
    # is_tmp가 True인 데이터만 조회
    results = db.query(PreprocessingEntity).filter(PreprocessingEntity.is_tmp == True).all()

    if len(results) < 10:
        raise Exception("Not enough files to process")

    tmp_dir = "/home/ubuntu/data/disrupt/tmp/"
    train_dir = "/home/ubuntu/data/disrupt/train/"
    test_dir = "/home/ubuntu/data/disrupt/test/"
    
    for idx, result in enumerate(results):
        # 파일 경로
        file_path = os.path.join(tmp_dir, result.npz_url)

        # 파일이 8개는 train 디렉토리로, 나머지 2개는 test 디렉토리로 이동
        if idx < 8:  # train 디렉토리로 이동
            target_dir = train_dir
        else:  # test 디렉토리로 이동
            target_dir = test_dir
        
        # 버전 이름 결정 (해당 디렉토리의 최신 버전 찾기)
        ver_name = get_next_version(target_dir)
        
        # 파일을 target 디렉토리로 이동하고, 새 경로 반환
        new_file_path = move_file(file_path, target_dir, ver_name)

        # DB에서 is_tmp를 False로 업데이트 및 npz_url 업데이트
        result.is_tmp = False
        result.npz_url = os.path.relpath(new_file_path, '/home/ubuntu/data/disrupt/')
        result.now_ver = int(ver_name[3:])
        db.commit()

    return {"message": "Files moved and database updated successfully"}
def add_weight_to_bbox(weight_map, bbox):
    x_min, y_min, x_max, y_max = bbox
    y_min, y_max, x_min, x_max = np.round([y_min, y_max, x_min, x_max]).astype(int)
    weight_map[y_min:y_max, x_min:x_max] = 1.0  # bbox에 가중치 1 부여
    return weight_map


def add_weight_to_landmarks(weight_map, landmarks):
    if len(landmarks) > 0:
        for (x, y, _) in landmarks:
            x, y = int(x), int(y)
            weight_map[max(0, y-5):min(weight_map.shape[0], y+5), max(0, x-5):min(weight_map.shape[1], x+5)] = 2.0  # 랜드마크 주변에 가중치 2 부여
    return weight_map


def create_preprocessing(db: Session, npz_url: str, is_tmp: bool = True, now_ver: int = None):
    db_item = PreprocessingEntity(npz_url=npz_url, is_tmp=is_tmp, now_ver=now_ver)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

    # @staticmethod
    # def update_user_predict(feedback_id: int, user_feedback: int, user_label: int):
    #     if user_feedback not in (0, 1) or user_label not in (0, 1):
    #         raise ValueError("userFeedback and userLabel must be 0 or 1.")

    #     with SessionLocal() as db:
    #         feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    #         if not feedback:
    #             raise Exception(f"Feedback not found with ID: {feedback_id}")

    #         feedback.user_predict = user_feedback
    #         feedback.user_label = user_label
    #         db.commit()
    #         db.refresh(feedback)

    #     return feedback
