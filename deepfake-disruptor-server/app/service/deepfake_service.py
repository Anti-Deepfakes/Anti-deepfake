from fastapi import HTTPException
from service.image_processing import save_image
import os
import logging
import traceback
import torch
import cv2
from io import BytesIO
import base64
import insightface

async def generate_deepfake_v1(face_detector, deepfake, image):
    image_path = await save_image(image)
    # print(deepfake)
    try:
        img1 = cv2.imread(image_path)
        img2 = cv2.imread("sample.png")

        face1 = face_detector.get(img1)[0]
        face2 = face_detector.get(img2)[0]

        final_image = deepfake.get(img2, face2, face1, paste_back=True)
        
        _, buffer = cv2.imencode('.jpg', final_image)
        img_byte_arr = buffer.tobytes()
        # print(final_image)

        # cv2.imwrite("generated.jpg", final_image)

        os.remove(image_path)

    except Exception as e:
        os.remove(image_path)
        logging.error("Exception occurred", exc_info=True)  # 로그에 예외 정보를 출력
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="딥페이크 생성이 불가합니다.")

    return base64.b64encode(img_byte_arr).decode('utf-8')

