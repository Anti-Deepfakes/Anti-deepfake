import os
import torch
from app.utils.train import train
from app.utils.dataloader import create_dataloader
from app.utils.hparams import HParam
from sqlalchemy.orm import Session  # Session 임포트 추가
from app.utils.mlflow_utils import init_mlflow, start_mlflow_run

def execute_training(train_path: str, test_path: str, data_version: int, config_path: str, checkpoint_path: str, db: Session):
    """
    학습을 실행하는 함수. save_dir 대신 MLflow로 모델을 관리.
    """
    print("[LOG: execute_training] Training execution started.")
    print(f"[LOG: execute_training] Parameters: train_path={train_path}, test_path={test_path}, data_version={data_version}, config_path={config_path}, checkpoint_path={checkpoint_path}")

    try:
        # 하이퍼파라미터 로드
        print("[LOG: execute_training] Loading hyperparameters.")
        hp = HParam(config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            hp_str = ''.join(f.readlines())
        print(f"[LOG: execute_training] Hyperparameters loaded successfully: {hp_str}")

        # 체크포인트 설정
        chkpt_path = checkpoint_path if checkpoint_path is not None else None
        print(f"[LOG: execute_training] Checkpoint path set to: {chkpt_path if chkpt_path else 'None'}")

        # 멀티 프로세싱 설정
        # print("[LOG: execute_training] Configuring multiprocessing settings.")
        # torch.set_num_threads(16)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "7"

        # import multiprocessing
        # multiprocessing.set_start_method('spawn')
        # print("[LOG: execute_training] Multiprocessing configured successfully.")

        # 데이터 로더 생성
        print("[LOG: execute_training] Creating data loaders.")
        # train_path, test_path
        train_loader, test_loader = create_dataloader(hp, train_path, test_path, data_version, db)
        print("[LOG: execute_training] Data loaders created successfully.")

        # 최신 버전 확인 및 새 버전 설정
        base_dir = "/home/ubuntu/data"
        model_type = "disrupt"
        print("[LOG: execute_training] Determining the latest version.")
        # version = get_latest_version(1) + 1
        version = data_version + 1 # 일단 이렇게
        print(f"[LOG: execute_training] New version set to: {version}")

        # 체크포인트 경로 설정
        version_str = f"ver{version:03d}"
        save_dir = os.path.join(base_dir, model_type, "model", f"{version_str}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"[LOG: execute_training] Save directory created: {save_dir}")

        chkpt_path = checkpoint_path if checkpoint_path else None

        # MLflow 초기화
        print("[LOG: execute_training] Initializing MLflow.")
        init_mlflow(experiment_name=f"{model_type}_experiment")
        mlflow_run = start_mlflow_run(run_name=f"{model_type}_train")
        print("[LOG: execute_training] MLflow initialized successfully.")

        # 학습 시작
        print("[LOG: execute_training] Starting training process.")
        train(
            hp=hp,
            train_loader=train_loader,
            valid_loader=test_loader,  # Note: Assuming test_loader is used as valid_loader
            chkpt_path=chkpt_path,
            save_dir=save_dir,
            db=db,
            version=version,
            data_version=data_version
        )
        print("[LOG: execute_training] Training process completed successfully.")

    except Exception as e:
        print(f"[ERROR: execute_training] An error occurred during training: {str(e)}")
        db.rollback()  # 오류 발생 시 롤백
        raise e
