from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from app.db_models import PreprocessingEntity

def create_dataloader(hp, train_path, test_path, data_version, db):
    """
    데이터 로더 생성 함수
    """
    print(f"[LOG: create_dataloader] Initializing data loader with:")
    print(f"  - train_path: {train_path}")
    print(f"  - test_path: {test_path}")

    # train 및 test 데이터 경로 설정
    print(f"[LOG: create_dataloader] Train set path: {train_path}")
    print(f"[LOG: create_dataloader] Test set path: {test_path}")

    # DB에서 train 및 test 데이터 가져오기
    train_image_paths = [
        os.path.join(train_path, row.npz_url)
        for row in db.query(PreprocessingEntity)
        .filter_by(now_ver=data_version, is_tmp=False)
        .filter(PreprocessingEntity.npz_url.startswith(train_path))
        .all()
    ]
    print(f"[LOG: create_dataloader] Loaded {len(train_path)} training paths from DB.")

    test_image_paths = [
        os.path.join(test_path, row.npz_url)
        for row in db.query(PreprocessingEntity)
        .filter_by(now_ver=data_version, is_tmp=False)
        .filter(PreprocessingEntity.npz_url.startswith(test_path))
        .all()
    ]
    print(f"[LOG: create_dataloader] Loaded {len(test_path)} testing paths from DB.")

    if not train_image_paths:
        raise ValueError(f"[ERROR: create_dataloader] No training data found for version {data_version} in the database.")
    if not test_image_paths:
        raise ValueError(f"[ERROR: create_dataloader] No testing data found for version {data_version} in the database.")

    # CustomDataset 인스턴스 생성
    train_set = CustomDataset(train_image_paths, hp)
    test_set = CustomDataset(test_image_paths, hp)
    print("[LOG: create_dataloader] CustomDataset instances created for train and test data.")

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=hp.train.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    print("[LOG: create_dataloader] DataLoaders created successfully.")

    # 추가: train_loader 데이터 확인
    print("[LOG: create_dataloader] Inspecting train_loader data:")
    for batch_idx, (image_tensor, image_weighted, bbox_tensor, landmarks_tensor) in enumerate(train_loader):
        print(f"[LOG: create_dataloader] Batch {batch_idx + 1}:")
        print(f"  - Image Tensor Shape: {image_tensor.shape}")
        print(f"  - Image Weighted Shape: {image_weighted.shape}")
        print(f"  - Bounding Box Tensor: {bbox_tensor}")
        print(f"  - Landmarks Tensor: {landmarks_tensor}")

    return train_loader, test_loader


class CustomDataset(Dataset):
    def __init__(self, image_paths, hp):
        self.image_paths = image_paths
        self.hp = hp
        print(f"[LOG: CustomDataset] Initialized with {len(image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        print(f"[LOG: CustomDataset] Loading image data from: {img_path}")
        
        try:
            input_data = np.load(img_path, allow_pickle=True)
        except Exception as e:
            print(f"[ERROR: CustomDataset] Failed to load file {img_path}: {e}")
            raise

        image_normalized = input_data["image_normalized"]
        weight_map = input_data["weight_map"]
        bbox = input_data["bbox"]
        landmarks = input_data["landmarks"]

        image_tensor = torch.tensor(image_normalized).float().cuda()
        weight_map_tensor = torch.tensor(weight_map).float().cuda()
        image_weighted = image_tensor + torch.stack([weight_map_tensor] * 3)

        bbox_tensor = torch.tensor(bbox).float().cuda()
        landmarks_tensor = torch.tensor(landmarks).float().cuda()

        print(f"[LOG: CustomDataset] Successfully loaded and processed data for index {idx}.")
        return image_tensor, image_weighted, bbox_tensor, landmarks_tensor

    def add_weight_to_bbox(self, weight_map, bbox):
        print("[LOG: CustomDataset] Adding weight to bounding box.")
        x_min, y_min, x_max, y_max = bbox
        weight_map[y_min:y_max, x_min:x_max] = 1.0  # bbox에 가중치 1 부여
        return weight_map

    def add_weight_to_landmarks(self, weight_map, landmarks):
        print("[LOG: CustomDataset] Adding weight to landmarks.")
        if landmarks:
            for (x, y) in landmarks:
                x, y = int(x), int(y)
                weight_map[max(0, y-5):min(weight_map.shape[0], y+5), max(0, x-5):min(weight_map.shape[1], x+5)] = 2.0  # 랜드마크 주변에 가중치 2 부여
        return weight_map
