import argparse
import glob
import cv2
import numpy as np
from torch.utils.data import random_split
from insightface.app import FaceAnalysis

def add_weight_to_bbox(weight_map, bbox):
    # print(bbox)
    x_min, y_min, x_max, y_max = bbox
    y_min, y_max, x_min, x_max = np.round([y_min, y_max, x_min, x_max]).astype(int)
    weight_map[y_min:y_max, x_min:x_max] = 1.0  # bbox에 가중치 1 부여
    return weight_map

def add_weight_to_landmarks(weight_map, landmarks):
    if len(landmarks) > 0:
        # print(landmarks[0].shape)
        for (x, y, _) in landmarks:
            x, y = int(x), int(y)
            weight_map[max(0, y-5):min(weight_map.shape[0], y+5), max(0, x-5):min(weight_map.shape[1], x+5)] = 2.0  # 랜드마크 주변에 가중치 2 부여
    return weight_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, required=False, default='')
    parser.add_argument('-r', '--train_ratio', type=float, required=False, default=0.7)
    args = parser.parse_args()

    total_data_list = glob.glob(args.data_dir + "/*.jpg", recursive=True)
    
    face_detector = FaceAnalysis(name='buffalo_l')
    face_detector.prepare(ctx_id=1, det_size=(224, 224))

    for img_name in total_data_list:
        file_name = img_name.replace("jpg", "npz")

        image = cv2.imread(img_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (224, 224))

        image_normalized = image_resized.astype(np.float32) / 255.0
        image_normalized = np.transpose(image_normalized, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        faces = face_detector.get(image)
        
        # bbox = np.zeros(4)
        # landmarks = np.zeros((68, 2))
        try:
            face = faces[0]
            bbox = face['bbox']
            landmarks = face['landmark_3d_68']
        except Exception as e:
            print(e, " : ", img_name)
            bbox = [0, 0, 0, 0]
            landmarks = []

        weight_map = np.zeros(image_normalized.shape[1:], dtype=np.float32)
        weight_map = add_weight_to_bbox(weight_map, bbox)
        weight_map = add_weight_to_landmarks(weight_map, landmarks)
        '''
        data_to_save = {
            "image_normalized": image_normalized,
            "weight_map": weight_map,
            "bbox": np.array(bbox, dtype=np.float32),
            "landmarks": np.array(landmarks, dtype=np.float32) if len(landmarks) > 0 else np.zeros((68, 3), dtype=np.float32)
        }
        '''
        np.savez(file_name, image_normalized=image_normalized, weight_map = weight_map, bbox=np.array(bbox, dtype=np.float32), 
                lansmarks=np.array(landmarks, dtype=np.float32) if len(landmarks) > 0 else np.zeros((68, 3), dtype=np.float32))

    train_size = int(args.train_ratio * len(total_data_list))
    test_size = len(total_data_list) - train_size
    train_set, test_set = random_split(total_data_list, [train_size, test_size])
    print(test_set)

    with open('test_set_images.ctrl', 'w') as f:
        for img_name in test_set:
            print(img_name)
            f.write(img_name.replace("jpg", "npz") + '\n')

    with open('train_set_images.ctrl', 'w') as f:
        for img_name in train_set:
            print(img_name)
            f.write(img_name.replace("jpg", "npz") + '\n')
