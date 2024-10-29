import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, default_collate
import cv2
from torchvision import transforms
from tqdm import tqdm
import wandb
import timm
import dlib

# Initialize wandb
wandb.init(project="deepfake_detection_multimodal")

# dlib의 얼굴 랜드마크 모델 로드 (사전 학습된 모델 필요)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # 사전 학습된 모델 경로

# 얼굴 부위 감지 및 분리 함수
def extract_facial_regions(image):
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

# Custom Dataset for loading images from folders
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Load real images
        for filename in os.listdir(real_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                self.image_paths.append(os.path.join(real_dir, filename))
                self.labels.append(0)  # 0 for real

        # Load fake images
        for filename in os.listdir(fake_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                self.image_paths.append(os.path.join(fake_dir, filename))
                self.labels.append(1)  # 1 for fake

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        # Load image in RGB format
        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract facial regions
        eye, nose, mouth, fallback = extract_facial_regions(image)
        if eye is None or nose is None or mouth is None:
            if fallback is not None:
                # 눈, 코, 입이 모두 감지되지 않을 때 전체 얼굴로 대체
                eye = fallback
                nose = fallback
                mouth = fallback
            else:
                return None

        if self.transform:
            eye = self.transform(eye)
            nose = self.transform(nose)
            mouth = self.transform(mouth)

        return (eye, nose, mouth), label

# 모델 정의 (눈, 코, 입 각각 적용)
def get_multimodal_model():
    # 세 개의 InceptionResNetV2 모델 초기화
    model_eye = timm.create_model('inception_resnet_v2', pretrained=True)
    model_nose = timm.create_model('inception_resnet_v2', pretrained=True)
    model_mouth = timm.create_model('inception_resnet_v2', pretrained=True)

    # 마지막 레이어 수정
    for model in [model_eye, model_nose, model_mouth]:
        model.classif = nn.Sequential(
            nn.Linear(model.classif.in_features, 128),  # 임베딩 크기 조정
            nn.ReLU()
        )

    # 최종 병합 레이어
    final_layer = nn.Sequential(
        nn.Linear(128 * 3, 1),
        nn.Sigmoid()
    )

    return model_eye, model_nose, model_mouth, final_layer

# 멀티모달 예측 함수
def multimodal_forward(eye, nose, mouth, model_eye, model_nose, model_mouth, final_layer):
    # 각 모델에 입력
    eye_features = model_eye(eye)
    nose_features = model_nose(nose)
    mouth_features = model_mouth(mouth)

    # 특징 결합
    combined_features = torch.cat((eye_features, nose_features, mouth_features), dim=1)

    # 최종 분류
    output = final_layer(combined_features)
    return output

# Training function
def train_model(models, final_layer, criterion, optimizer, dataloader, device):
    model_eye, model_nose, model_mouth = models
    model_eye.train()
    model_nose.train()
    model_mouth.train()
    final_layer.train()

    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        if batch is None:
            continue  # batch가 None인 경우 건너뜀

        images, labels = batch
        if images is None or labels is None:
            continue  # 얼굴 부위가 없거나 이미지가 None인 경우 건너뜀

        eye, nose, mouth = images
        eye = eye.to(device).float()
        nose = nose.to(device).float()
        mouth = mouth.to(device).float()
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = multimodal_forward(eye, nose, mouth, model_eye, model_nose, model_mouth, final_layer)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * eye.size(0)
        wandb.log({"train_loss": loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# collate_fn 정의
def collate_fn(batch):
    # None 값을 필터링하여 유효한 데이터만 남기기
    batch = [data for data in batch if data is not None]
    if len(batch) == 0:
        return None  # 모든 데이터가 None인 경우를 처리
    return default_collate(batch)

# Main execution
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="멀티모달 딥페이크 탐지 모델 학습 스크립트")
    parser.add_argument("--epochs", type=int, default=10, help="학습 반복 횟수 (기본값: 10)")
    parser.add_argument("--batch_size", type=int, default=4, help="훈련 배치 크기 (기본값: 4)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="학습률 (기본값: 0.001)")
    parser.add_argument("--real_dir", type=str, default="./data/REAL", help="실제 이미지가 저장된 디렉토리 경로 (기본값: ./data/REAL)")
    parser.add_argument("--fake_dir", type=str, default="./data/FAKE", help="가짜 이미지가 저장된 디렉토리 경로 (기본값: ./data/FAKE)")
    parser.add_argument("--gpu", type=int, default=0, help="학습에 사용할 GPU ID (기본값: 0)")
    args = parser.parse_args()

    # Check if the specified GPU ID is available
    if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print(f"Invalid GPU ID {args.gpu}. Falling back to CPU.")
    print(f"Using device: {device}")

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299)),  # Resize to match InceptionResNetV2 input size
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Create dataset and dataloaders
    dataset = DeepfakeDataset(args.real_dir, args.fake_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize models
    model_eye, model_nose, model_mouth, final_layer = get_multimodal_model()
    models = (model_eye.to(device), model_nose.to(device), model_mouth.to(device))
    final_layer = final_layer.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(list(model_eye.parameters()) +
                           list(model_nose.parameters()) +
                           list(model_mouth.parameters()) +
                           list(final_layer.parameters()), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_model(models, final_layer, criterion, optimizer, train_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss_epoch": train_loss})

    # Save the trained models
    torch.save({
        'model_eye_state_dict': model_eye.state_dict(),
        'model_nose_state_dict': model_nose.state_dict(),
        'model_mouth_state_dict': model_mouth.state_dict(),
        'final_layer_state_dict': final_layer.state_dict()
    }, "deepfake_detector_multimodal.pth")

    wandb.finish()
