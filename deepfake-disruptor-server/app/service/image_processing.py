import os
import numpy as np
import cv2
import torch
from fastapi import HTTPException
import torch.nn.functional as F

async def save_image(image):
    try:
        os.makedirs("./data", exist_ok=True)
        image_path = f"./data/{image.filename}"
        # print(image_path)
        contents = await image.read()
        # print(contents)
        with open(image_path, "wb") as f:
            f.write(contents)

    except Exception as e:
        raise HTTPException(status_code=500, detail="이미지 처리가 되지 않았습니다.")

    return image_path

async def preprocessing(image_path, face_detector):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    max_dimension = max(height, width)

    if max_dimension > 1000:
        scale_ratio = 1000 / max_dimension
        
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)
        
        image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_normalized = np.transpose(image_normalized, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    
    faces = face_detector.get(image)
    # print(faces)
    # print(F.softmax
    # print(faces.shape)
    
    try:
        face = faces[0]
        bbox = face['bbox']
        landmarks = face['landmark_3d_68']
        
    except Exception as e:
        print(e, " : ", img_name)
        
        bbox = [0, 0, 0, 0]
        landmarks = []

    weight_map = np.zeros(image_normalized.shape[1:], dtype=np.float32)
    weight_map = await add_weight_to_bbox(weight_map, bbox)
    weight_map = await add_weight_to_landmarks(weight_map, landmarks)
    
    lansmarks=np.array(landmarks, dtype=np.float32) if len(landmarks) > 0 else np.zeros((68, 3), dtype=np.float32)
   
    image_tensor = torch.tensor(image_normalized).float().cuda()
    
    weight_map_tensor = torch.tensor(weight_map).float().cuda()
    
    image_weighted = image_tensor + torch.stack([weight_map_tensor] * 3)

    return image_tensor.unsqueeze(0), image_weighted.unsqueeze(0), torch.stack([weight_map_tensor]*3).unsqueeze(0)


async def add_weight_to_bbox(weight_map, bbox):
    x_min, y_min, x_max, y_max = bbox
    y_min, y_max, x_min, x_max = np.round([y_min, y_max, x_min, x_max]).astype(int)
    weight_map[y_min:y_max, x_min:x_max] = 1.0  # bbox에 가중치 1 부여
    return weight_map


async def add_weight_to_landmarks(weight_map, landmarks):
    if len(landmarks) > 0:
        for (x, y, _) in landmarks:
            x, y = int(x), int(y)
            weight_map[max(0, y-5):min(weight_map.shape[0], y+5), max(0, x-5):min(weight_map.shape[1], x+5)] = 2.0  # 랜드마크 주변에 가중치 2 부여
    return weight_map

async def tensor_to_cv2_image(tensor_image):
    image_np = tensor_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255.0
    image_np = image_np.astype(np.uint8)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2
