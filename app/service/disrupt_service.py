from fastapi import HTTPException
from service.image_processing import save_image
import os
import logging
import traceback
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from io import BytesIO
import base64

async def generate_disrupt_v1(model, image, device):
    # print(model)
    # print(image)
    # print(device)
   
    image_path = await save_image(image)
    # print(image_path)
    try:
        input_img_size, input_image = await open_image(image_path, device)
        with torch.no_grad():
            perturbations = model(input_image)
            # print(perturbations)
            perturbed_images = input_image + perturbations

            final_image = await tensor_to_image(perturbed_images.squeeze(0))
            
            resized_img = final_image.resize(input_img_size, Image.LANCZOS)

            img_byte_arr = BytesIO()
            resized_img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)

            # resized_img.save("./disrupt_image.jpg")
        os.remove(image_path)

    except Exception as e:
        os.remove(image_path)
        logging.error("Exception occurred", exc_info=True)  # 로그에 예외 정보를 출력
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="사진 변환이 불가합니다.")

    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')


async def open_image(image_path, device):
    ori_image = Image.open(image_path).convert("RGB")
    ori_image_size = ori_image.size
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    ori_image = transform(ori_image).unsqueeze(0).to(device)
    return ori_image_size, ori_image


async def tensor_to_image(tensor):
    to_pil = ToPILImage()
    return to_pil(tensor.cpu().detach())
