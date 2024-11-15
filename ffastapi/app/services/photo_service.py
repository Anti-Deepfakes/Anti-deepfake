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

            # # 확률이 유효할 때만 DB에 저장
            # if deepfake_probability != -1:
            #     with SessionLocal() as db:
            #         feedback = Feedback(
            #             probability=1 if deepfake_probability >= 50 else 0,
            #             user_label=2,
            #             user_predict=2
            #         )
            #         db.add(feedback)
            #         db.commit()
            #         db.refresh(feedback)

            #         result["deepfakeProbability"] = deepfake_probability
            #         result["feedbackId"] = feedback.id

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

        # 이미지도 동일한 이름으로 저장
        cv2.imwrite(os.path.join(directory, f"{file_name}.jpg"), image_resized)

        print(f"Saved {file_name}.npz and {file_name}.jpg")
        print("end saving")
        print("start Database Save")
        npz_url = os.path.join(directory, f"{file_name}.npz")
        create_preprocessing(db, npz_url,True,None)
        # 반환값으로 파일명 또는 성공 메시지를 반환
        return {"message": "Preprocessing successful", "filename": f"{file_name}.npz"}
    

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
