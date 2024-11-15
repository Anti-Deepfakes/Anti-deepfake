from fastapi import APIRouter, UploadFile, File, HTTPException,Depends
from app.services.photo_service import PhotoService
from db import get_db  # database.py에서 get_db 임포트
from sqlalchemy.orm import Session

router = APIRouter(
    prefix="/photos",
    tags=["photos"]
)

@router.post("/detection")
async def detect(image: UploadFile = File(...)):
    try:
        response = PhotoService.file_upload(image)  # 동기식 호출
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during file upload: {str(e)}")
    
@router.post("/preprocessing")
async def detect(image: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        response = PhotoService.preprocessing(image,db)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {str(e)}")
    


# @router.put("/feedback")
# async def put_feedback(feedback_id: int, user_feedback: int, user_label: int):
#     try:
#         feedback = PhotoService.update_user_predict(feedback_id, user_feedback, user_label)
#         return feedback
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
