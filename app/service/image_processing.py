import os
from fastapi import HTTPException


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
