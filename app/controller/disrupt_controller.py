from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from trainer import execute_training
from utils.evaluation import evaluate_and_save_performance
from utils.mlflow_utils import get_latest_version
from utils.db import get_db


router = APIRouter()

@router.post("/train")
async def train_disrupt_model(
    data_dir: str = "/home/ubuntu/attack/DeepFake_Disruptor/data/img_align_celeba/img_align_celeba",
    data_version: int = 0,
    config_path: str = "./config/config.yaml",
    checkpoint_path: str = None,
    db: Session = Depends(get_db)
):
    """
    학습을 트리거하는 엔드포인트.
    """
    try:
        execute_training(data_dir, data_version, config_path, checkpoint_path, db)

        # 여기에 학습한 모델 성능평가지표랑 현재 배포중인 모델 성능평가지표 비교 후 있으면 바꾸는 로직 추가!

        return {"message": "Training started successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
