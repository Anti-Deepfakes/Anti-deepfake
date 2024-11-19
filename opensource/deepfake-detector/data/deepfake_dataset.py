# datasets/deepfake_dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset
from skimage.restoration import denoise_wavelet
from torchvision import transforms

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Extract noise residual
        noise_residual = image - denoise_wavelet(image, channel_axis=-1, convert2ycbcr=False)

        if self.transform:
            noise_residual = self.transform(noise_residual)

        return noise_residual, label
