from fastapi import HTTPException
from service.image_processing import save_image, extract_facial_regions
import os
import logging
import traceback
import torch
import cv2
from torchvision import transforms

# Transform 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((299, 299)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

async def ai_inference(models, image, device):
    model_eye, model_nose, model_mouth, final_layer = models

    # 이미지 저장
    image_path = await save_image(image)

    try:
        # 이미지 로드 및 전처리
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 얼굴 부위 추출
        eye, nose, mouth, fallback = extract_facial_regions(image_rgb)

        if fallback is not None:
            eye, nose, mouth = fallback, fallback, fallback

        # 얼굴 부위가 모두 감지되었는지 확인
        if eye is None or nose is None or mouth is None:
            raise HTTPException(status_code=400, detail="얼굴 부위를 감지할 수 없습니다.")

        # Transform 적용
        eye_tensor = transform(eye).unsqueeze(0).to(device)
        nose_tensor = transform(nose).unsqueeze(0).to(device)
        mouth_tensor = transform(mouth).unsqueeze(0).to(device)

        # 예측 수행
        with torch.no_grad():
            eye_features = model_eye(eye_tensor)
            nose_features = model_nose(nose_tensor)
            mouth_features = model_mouth(mouth_tensor)
            combined_features = torch.cat((eye_features, nose_features, mouth_features), dim=1)
            output = final_layer(combined_features)
            probability = torch.sigmoid(output).item()

        return [{
            "label": "Deepfake",
            "conf": round(probability * 100, 2)
        }]
    except Exception as e:
        logging.error("Inference failed", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="모델 예측에 실패했습니다.")
    finally:
        # 저장된 이미지 삭제
        if os.path.exists(image_path):
            os.remove(image_path)