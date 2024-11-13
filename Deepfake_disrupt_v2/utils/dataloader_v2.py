from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def create_dataloader(hp, data_path, face_detector):
    # transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Resize([hp.data.image_size, hp.data.image_size])
    # ])

    with open('train_set_images.ctrl', 'r') as f:
        train_image_paths = [line.strip() for line in f if line.strip()]

    # test_set_images.ctrl 파일에서 경로 읽기
    with open('test_set_images.ctrl', 'r') as f:
        test_image_paths = [line.strip() for line in f if line.strip()]

    # CustomDataset 인스턴스 생성
    train_set = CustomDataset(train_image_paths, hp, face_detector)
    test_set = CustomDataset(test_image_paths, hp, face_detector)

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=hp.train.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=hp.train.batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader


class CustomDataset(Dataset):
    def __init__(self, image_paths, hp, face_detector):
        self.image_paths = image_paths
        self.hp = hp
        self.face_detector = face_detector
        # self.face_detector = FaceAnalysis(name='buffalo_l')
        # self.face_detector.prepare(ctx_id=0, det_size = (hp.data.image_size, hp.data.image_size))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        print(img_path)

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.hp.data.image_size, self.hp.data.image_size))

        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = np.transpose(image_normalized, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        faces = self.face_detector.get(image)

        try:
            face = faces[0]
            bbox = face['bbox']
            landmarks = face['landmarks']
        except Exception as e:
            print(e, " : ", img_path)
            bbox = [0, 0, 0, 0]
            landmarks = []
        
        weight_map = np.zeros(image_normalized.shape[1:], dtype=np.float32)
        weight_map = self.add_weight_to_bbox(weight_map, bbox)
        weight_map = self.add_weight_to_landmarks(weight_map, landmarks)

        image_tensor = torch.tensor(image_normalized).float().cuda()
        weight_map_tensor = torch.tensor(weight_map).float().cuda()

        image_weighted = image_tensor + torch.stack([weight_map_tensor] * 3)

        bbox_tensor = torch.tensor(bbox).float().cuda()
        landmarks_tensor = torch.tensor(landmarks).float().cuda() if landmarks else torch.zeros([68, 2]).cuda()

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
