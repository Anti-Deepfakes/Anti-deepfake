from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

def create_dataloader(hp, data_dir, data_version):
    # transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Resize([hp.data.image_size, hp.data.image_size])
    # ])

    # data_version에 따라 디렉토리 설정
    versioned_data_dir = os.path.join(data_dir, f"ver_{data_version}")

    # train 및 test 데이터 경로 설정
    train_set_path = os.path.join(data_dir, f"disrupt/train/ver_{data_version}")
    test_set_path = os.path.join(data_dir, f"disrupt/test/ver_{data_version}")

    # DB에서 train 및 test 데이터 가져오기
    train_image_paths = [
        os.path.join(data_dir, row.npz_url)
        for row in db.query(PreprocessingEntity)
        .filter_by(now_ver=data_version, is_tmp=False)
        .filter(PreprocessingEntity.npz_url.startswith(train_set_path))
        .all()
    ]

    test_image_paths = [
        os.path.join(data_dir, row.npz_url)
        for row in db.query(PreprocessingEntity)
        .filter_by(now_ver=data_version, is_tmp=False)
        .filter(PreprocessingEntity.npz_url.startswith(test_set_path))
        .all()
    ]

    if not train_image_paths:
        raise ValueError(f"No training data found for version {data_version} in the database.")
    if not test_image_paths:
        raise ValueError(f"No testing data found for version {data_version} in the database.")

    # CustomDataset 인스턴스 생성
    train_set = CustomDataset(train_image_paths, hp)
    test_set = CustomDataset(test_image_paths, hp)

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=hp.train.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, test_loader


class CustomDataset(Dataset):
    def __init__(self, image_paths, hp):
        self.image_paths = image_paths
        self.hp = hp
        # self.face_detector = face_detector
        # self.face_detector = FaceAnalysis(name='buffalo_l')
        # self.face_detector.prepare(ctx_id=0, det_size = (hp.data.image_size, hp.data.image_size))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # print(img_path)
        input_data = np.load(img_path, allow_pickle=True)
        # print(input_data.files)
        image_normalized = input_data["image_normalized"]
        weight_map = input_data["weight_map"]
        bbox = input_data["bbox"]
        landmarks = input_data["lansmarks"]

        image_tensor = torch.tensor(image_normalized).float().cuda()
        weight_map_tensor = torch.tensor(weight_map).float().cuda()

        image_weighted = image_tensor + torch.stack([weight_map_tensor] * 3)

        bbox_tensor = torch.tensor(bbox).float().cuda()
        # if landmarks.shape[-1] == 2:
        #    landmarks = np.zeros([68,3])
        landmarks_tensor = torch.tensor(landmarks).float().cuda()

        # image_concat = torch.cat([image_weighted, bbox_tensor.unsqueeze(0), landmarks_tensor.unsqueeze(0)], dim=1)
        
        return image_tensor, image_weighted, bbox_tensor, landmarks_tensor

    def add_weight_to_bbox(self, weight_map, bbox):
        x_min, y_min, x_max, y_max = bbox
        weight_map[y_min:y_max, x_min:x_max] = 1.0  # bbox에 가중치 1 부여
        return weight_map

    def add_weight_to_landmarks(self, weight_map, landmarks):
        if landmarks:
            for (x, y) in landmarks:
                x, y = int(x), int(y)
                weight_map[max(0, y-5):min(weight_map.shape[0], y+5), max(0, x-5):min(weight_map.shape[1], x+5)] = 2.0  # 랜드마크 주변에 가중치 2 부여
        return weight_map
