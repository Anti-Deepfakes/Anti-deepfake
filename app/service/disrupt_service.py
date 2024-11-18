from fastapi import HTTPException
from service.image_processing import save_image, preprocessing, tensor_to_cv2_image
import os
import logging
import traceback
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from io import BytesIO
import torch.nn.functional as F
import base64
import cv2
import torch.backends.cudnn as cudnn
import torch
import torch.fft
import numpy as np

async def generate_disrupt_v1(model, image, device, face_detector):
    image_path = await save_image(image)
      

    image_tensor, image_weighted, weighted_map = await preprocessing(image_path, face_detector)

    try:

        with torch.no_grad():
            
            perturbations = model(image_weighted)
            perturbations = F.interpolate(perturbations, size=(image_tensor.shape[2], image_tensor.shape[3]), mode='area')
            print(image_tensor)
            
            # target_rgb = (255, 204, 153)
            # perturbations[:, 0, ...] *= torch.tensor(target_rgb[0] / 255.0).float().cuda()
            # perturbations[:, 1, ...] *= torch.tensor(target_rgb[1] / 255.0).float().cuda()
            # perturbations[:, 2, ...] *= torch.tensor(target_rgb[2] / 255.0).float().cuda()

            perturbed_images = image_tensor + (perturbations + (2-weighted_map)*0.25)
            perturbed_images += image_tensor

            min_val = perturbed_images.min()
            max_val = perturbed_images.max()

            perturbed_images = (perturbed_images - min_val) / (max_val - min_val)

            # perturbed_images += image_tensor
            # min_val = perturbed_images.min()
            # max_val = perturbed_images.max()

            # perturbed_images = image_tensor * (perturbations-1)
            print(perturbed_images)
            # perturbed_images = torch.clamp(perturbed_images, min=0.0, max=1.0)

            final_image = await tensor_to_cv2_image(perturbed_images)
            
            final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(final_image_rgb)

            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            cv2.imwrite("./disrupt_image.jpg", final_image)
        os.remove(image_path)

    except Exception as e:
        os.remove(image_path)
        logging.error("Exception occurred", exc_info=True)  # 로그에 예외 정보를 출력
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="사진 변환이 불가합니다.")

    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

