from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.utils.trainer import execute_training
from app.utils.evaluation import evaluate_and_save_performance
from app.utils.mlflow_utils import get_latest_version
from app.utils.db import get_db

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
    # 요청 시작 시 로그
    print(f"[LOG: train_disrupt_model] Received request to trigger training.")
    print(f"  - Input Parameters:")
    print(f"    - data_dir: {data_dir}")
    print(f"    - data_version: {data_version}")
    print(f"    - config_path: {config_path}")
    print(f"    - checkpoint_path: {checkpoint_path if checkpoint_path else 'None'}")

    try:
        # 학습 실행
        print("[LOG: train_disrupt_model] Starting execute_training function...")
        execute_training(data_dir, data_version, config_path, checkpoint_path, db)
        print("[LOG: train_disrupt_model] execute_training function completed successfully.")

        # 성능 평가 후 모델 업데이트 로직 (향후 추가)
        print("[LOG: train_disrupt_model] Placeholder for model performance evaluation logic.")

        # 요청 성공 로그
        print("[LOG: train_disrupt_model] Training triggered successfully. Returning response.")
        return {"message": "Training started successfully."}

    except Exception as e:
        # 에러 발생 시 로그
        print(f"[ERROR: train_disrupt_model] Exception occurred during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")