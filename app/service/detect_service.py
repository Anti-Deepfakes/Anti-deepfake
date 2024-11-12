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
    transforms.Resize((128, 1)),  # 모델이 기대하는 크기로 조정
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 모델에 맞춘 정규화
])


async def ai_inference(models, image, label_mapper):
    model_eye, model_nose, model_mouth, final_layer = models
    image_path = await save_image(image)

    outputs = []
    try:
        # 이미지에서 얼굴 부위를 추출
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        eye, nose, mouth, fallback = extract_facial_regions(image_rgb)

        # 얼굴에서 눈코입이 감지되지 않은 경우: 전체 얼굴을 'final_layer'만으로 예측
        if fallback is not None:
            fallback = transform(fallback)
            face_tensor = torch.tensor(fallback).unsqueeze(0).to(next(final_layer.parameters()).device)
            print(fallback.shape)
            print(face_tensor.shape)
            with torch.no_grad():
                combined_features = final_layer(face_tensor)
                probability = torch.sigmoid(combined_features).item()
            outputs.append({
                "label": "Deepfake",
                "conf": round(probability * 100, 2)
            })

        # 얼굴 부위(눈, 코, 입)가 모두 감지된 경우: 각각의 모델(eye, nose, mouth)을 사용하여 예측
        elif eye is not None and nose is not None and mouth is not None:
            eye = transform(eye)
            nose = transform(nose)
            mouth = transform(mouth)
            eye_tensor = torch.tensor(eye).unsqueeze(0).float().to(next(model_eye.parameters()).device)
            nose_tensor = torch.tensor(nose).unsqueeze(0).float().to(next(model_nose.parameters()).device)
            mouth_tensor = torch.tensor(mouth).unsqueeze(0).float().to(next(model_mouth.parameters()).device)
            print(next(model_eye.parameters()).device)

            # 예측 수행
            with torch.no_grad():
                eye_features = model_eye(eye_tensor)
                nose_features = model_nose(nose_tensor)
                mouth_features = model_mouth(mouth_tensor)
                combined_features = torch.cat((eye_features, nose_features, mouth_features), dim=1)
                output = final_layer(combined_features)
                probability = torch.sigmoid(output).item()

            outputs.append({
                "label": "Deepfake",
                "conf": round(probability * 100, 2)
            })
        
        # 예측이 완료되면 저장된 이미지를 삭제
        os.remove(image_path)

    except Exception as e:
        os.remove(image_path)
        logging.error("Exception occurred", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="모델 예측이 불가합니다.")

    return outputs
