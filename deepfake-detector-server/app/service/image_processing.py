import os
from fastapi import HTTPException
import cv2
import dlib

# 얼굴 감지 모델과 랜드마크 예측 모델 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")  # 파일 경로 수정 필요


async def save_image(image):
    try:
        os.makedirs("./data", exist_ok=True)
        image_path = f"./data/{image.filename}"
        contents = await image.read()
        with open(image_path, "wb") as f:
            f.write(contents)

    except Exception as e:
        raise HTTPException(status_code=500, detail="이미지 처리가 되지 않았습니다.")

    return image_path

def extract_facial_regions(image):
    """
    이미지에서 얼굴 부위(눈, 코, 입)를 감지하고 추출합니다.

    Args:
        image (numpy.ndarray): RGB 형식의 입력 이미지

    Returns:
        tuple: 눈, 코, 입 영역이 있는 이미지 부분 또는 전체 얼굴 이미지
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, None, None, image  # 얼굴이 감지되지 않은 경우 전체 얼굴 반환

    # 첫 번째 얼굴 선택
    face = faces[0]
    landmarks = predictor(gray, face)

    # 눈, 코, 입 좌표 추출
    eye_region = image[landmarks.part(36).y:landmarks.part(45).y, landmarks.part(36).x:landmarks.part(45).x]
    nose_region = image[landmarks.part(27).y:landmarks.part(35).y, landmarks.part(31).x:landmarks.part(35).x]
    mouth_region = image[landmarks.part(48).y:landmarks.part(57).y, landmarks.part(48).x:landmarks.part(54).x]

    # 만약 눈, 코, 입이 감지되지 않으면 전체 얼굴 반환
    if eye_region.size == 0 or nose_region.size == 0 or mouth_region.size == 0:
        return None, None, None, image

    return eye_region, nose_region, mouth_region, None
