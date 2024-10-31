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
from utils.face_utils import extract_facial_regions

# Initialize wandb
wandb.init(project="deepfake_detection_multimodal")


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): 검증 손실이 개선되지 않아도 기다리는 epoch 수
            min_delta (float): 개선으로 간주될 최소 변화량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 개선되면 카운터 초기화
        else:
            self.counter += 1  # 개선되지 않으면 카운터 증가
            if self.counter >= self.patience:
                self.early_stop = True  # 조기 종료 조건 충족


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


# 검증 함수 추가
def validate_model(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            if images is None or labels is None:
                continue
            eye, nose, mouth = images
            eye = eye.to(device).float()
            nose = nose.to(device).float()
            mouth = mouth.to(device).float()
            labels = labels.to(device).float().unsqueeze(1)

            outputs = multimodal_forward(eye, nose, mouth, *model)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * eye.size(0)

    val_loss = running_loss / len(val_loader.dataset)
    return val_loss


# 모델 체크포인트 저장 함수
def save_checkpoint(epoch, model_eye, model_nose, model_mouth, final_layer, checkpoint_dir="checkpoints", best=False,
                    max_checkpoints=10):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_eye_state_dict': model_eye.state_dict(),
        'model_nose_state_dict': model_nose.state_dict(),
        'model_mouth_state_dict': model_mouth.state_dict(),
        'final_layer_state_dict': final_layer.state_dict()
    }

    # Save checkpoint
    if best:
        filename = "best_model.pth"
    else:
        filename = f"model_checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    print(f"Checkpoint saved: {filename}")

    # Delete old checkpoints if exceeding max_checkpoints
    checkpoints = sorted(
        [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("model_checkpoint")],
        key=os.path.getctime
    )
    if len(checkpoints) > max_checkpoints:
        os.remove(checkpoints[0])
        print(f"Deleted old checkpoint: {checkpoints[0]}")


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
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu < torch.cuda.device_count() else "cpu")
    print(f"Using device: {device}")

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((299, 299)),
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

    # EarlyStopping 초기화
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    best_val_loss = float('inf')

    # Training loop with validation and early stopping
    for epoch in range(args.epochs):
        train_loss = train_model(models, final_layer, criterion, optimizer, train_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {train_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss_epoch": train_loss})

        # Validation and early stopping check
        val_loss = validate_model((model_eye, model_nose, model_mouth, final_layer), criterion, val_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "val_loss_epoch": val_loss})

        # Save checkpoint every epoch
        save_checkpoint(epoch + 1, model_eye, model_nose, model_mouth, final_layer)

        # Check if the current model is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch + 1, model_eye, model_nose, model_mouth, final_layer, best=True)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    wandb.finish()