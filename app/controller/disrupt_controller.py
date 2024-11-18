from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.utils.trainer import execute_training
from app.utils.db import get_db
from pydantic import BaseModel
import os

router = APIRouter()

# 요청 데이터 모델 정의
class TrainRequest(BaseModel):
    data_version: int = 0
    checkpoint_path: str = None

@router.post("/train")
async def train_disrupt_model(
    request: TrainRequest,  # 요청 데이터 매핑
    db: Session = Depends(get_db)
):
    """
    학습을 트리거하는 엔드포인트.
    """

    data_version = request.data_version
    checkpoint_path = request.checkpoint_path

    # 경로 설정
    if data_version == 0:
        train_path = "/home/ubuntu/attack/DeepFake_Disruptor/data/img_align_celeba/img_align_celeba"
        test_path = "/home/ubuntu/attack/DeepFake_Disruptor/data/img_align_celeba/img_align_celeba"
    else:
        version_str = f"ver{data_version:03d}"
        train_path = f"/home/ubuntu/data/disrupt/train/{version_str}"
        test_path = f"/home/ubuntu/data/disrupt/test/{version_str}"

    # config_path = "./config/config.yaml"
    config_path = "/workspace/app/config/config.yaml"

    # 요청 시작 시 로그
    print(f"[LOG: train_disrupt_model] Received request to trigger training.")
    print(f"  - Input Parameters:")
    print(f"    - data_version: {data_version}")
    print(f"    - train_path: {train_path}")
    print(f"    - test_path: {test_path}")
    print(f"    - config_path: {config_path}")
    print(f"[DEBUG] CUDA_VISIBLE_DEVICES in Python: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    try:
        # 학습 실행
        print("[LOG: train_disrupt_model] Starting execute_training function...")
        execute_training(train_path, test_path, data_version, config_path, checkpoint_path, db)
        print("[LOG: train_disrupt_model] execute_training function completed successfully.")

        # 성능 평가 후 모델 업데이트 로직 (향후 추가)
        print("[LOG: train_disrupt_model] 성능 평가 후 모델 업데이트 로직")

        # 새로 학습된 모델 성능 조회 (예: DB에서 데이터 가져오기)
        new_model_performance = get_new_model_performance(db, data_version)
        print(f"[LOG: train_disrupt_model] New model performance: {new_model_performance}")

        # 현재 배포된 모델 성능 조회
        deployed_model_performance = get_deployed_model_performance(db)
        print(f"[LOG: train_disrupt_model] Deployed model performance: {deployed_model_performance}")

        # 성능 비교
        if new_model_performance["accuracy"] > deployed_model_performance["accuracy"]:
            print("[LOG: train_disrupt_model] New model outperforms deployed model. Sending update trigger...")

            # 모델 업데이트 트리거
            deployment_server_url = "http://model-deployment-server/update"
            response = requests.post(
                deployment_server_url,
                json={
                    "model_version": data_version,
                    "model_path": f"/home/ubuntu/data/disrupt/model/ver{data_version:03d}/model_{data_version}.pth"
                },
                timeout=10
            )
            if response.status_code == 200:
                print("[LOG: train_disrupt_model] Model deployment updated successfully.")
            else:
                print(f"[ERROR: train_disrupt_model] Failed to update model deployment: {response.text}")
                raise HTTPException(status_code=500, detail="Model update failed.")

        else:
            print("[LOG: train_disrupt_model] New model does not outperform deployed model. No update triggered.")

        # 요청 성공 로그
        print("[LOG: train_disrupt_model] Training triggered successfully. Returning response.")
        return {"message": "Training started successfully."}

    except Exception as e:
        # 에러 발생 시 로그
        print(f"[ERROR: train_disrupt_model] Exception occurred during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


def get_new_model_performance(db: Session, data_version: int):
    """
    DB에서 새로 학습된 모델의 성능을 가져옵니다.
    """
    performance = db.query(Performance).filter(Performance.data_version == data_version).first()
    if not performance:
        raise ValueError(f"No performance data found for version {data_version}")
    return {
        "accuracy": performance.accuracy,
        "loss": performance.total_loss
    }


def get_deployed_model_performance(db: Session):
    """
    DB에서 현재 배포된 모델의 성능을 가져옵니다.
    """
    deployed_model = db.query(Performance).filter(Performance.is_deployed == True).first()
    if not deployed_model:
        raise ValueError("No deployed model performance data found")
    return {
        "accuracy": deployed_model.accuracy,
        "loss": deployed_model.total_loss
    }