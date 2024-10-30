import os

import argparse
import torch
import cv2
from torchvision import transforms
import timm
from utils.face_utils import extract_facial_regions

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((299, 299)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 모델 로드 함수
def load_multimodal_models(weights_path):
    # 각각의 모델과 최종 레이어 초기화 및 가중치 로드
    model_eye = timm.create_model('inception_resnet_v2', pretrained=True)
    model_nose = timm.create_model('inception_resnet_v2', pretrained=True)
    model_mouth = timm.create_model('inception_resnet_v2', pretrained=True)
    final_layer = torch.nn.Sequential(
        torch.nn.Linear(3000, 1),  # 입력 크기를 combined_features에 맞춰 조정
        torch.nn.Sigmoid()
    )

    model_eye.eval()
    model_nose.eval()
    model_mouth.eval()
    final_layer.eval()

    return model_eye, model_nose, model_mouth, final_layer


# 예측 함수
def predict(image_path, model_eye, model_nose, model_mouth, final_layer, device):
    # 이미지 로드 및 얼굴 부위 추출
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지를 로드할 수 없습니다.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 부위 추출
    eye, nose, mouth, fallback = extract_facial_regions(image_rgb)
    if eye is None or nose is None or mouth is None:
        if fallback is not None:
            eye, nose, mouth = fallback, fallback, fallback
        else:
            raise ValueError("얼굴 부위를 감지할 수 없습니다.")

    # 이미지 전처리
    eye = transform(eye).unsqueeze(0).to(device)
    nose = transform(nose).unsqueeze(0).to(device)
    mouth = transform(mouth).unsqueeze(0).to(device)

    # 예측 수행
    eye_features = model_eye(eye)
    nose_features = model_nose(nose)
    mouth_features = model_mouth(mouth)

    # 특징 결합 및 최종 예측
    combined_features = torch.cat((eye_features, nose_features, mouth_features), dim=1)
    output = final_layer(combined_features)

    # 결과 출력
    real_prob = output.item()
    fake_prob = 1 - real_prob
    return fake_prob, real_prob


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="딥페이크 탐지 모델 예측 스크립트")
    parser.add_argument("--image_path", type=str, required=True, help="예측할 이미지의 경로")
    parser.add_argument("--weights_path", type=str, required=True, help="모델 가중치가 저장된 디렉토리 경로")
    parser.add_argument("--gpu", type=int, default=0, help="사용할 GPU ID (기본값: 0)")
    args = parser.parse_args()

    # 장치 설정
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu < torch.cuda.device_count() else "cpu")
    print(f"Using device: {device}")

    # 모델 로드
    model_eye, model_nose, model_mouth, final_layer = load_multimodal_models(args.weights_path)
    model_eye, model_nose, model_mouth, final_layer = model_eye.to(device), model_nose.to(device), model_mouth.to(
        device), final_layer.to(device)

    # 예측 수행
    fake_prob, real_prob = predict(args.image_path, model_eye, model_nose, model_mouth, final_layer, device)
    print(f"딥페이크 확률: {fake_prob * 100:.2f}%, 실제 확률: {real_prob * 100:.2f}%")
