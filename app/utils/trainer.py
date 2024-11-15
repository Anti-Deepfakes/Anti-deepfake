import os
import torch
from utils.train import train
from utils.dataloader import create_dataloader
from utils.hparams import HParam

def execute_training(data_dir: str, data_version: int, config_path: str, checkpoint_path: str, db: Session):
    """
    학습을 실행하는 함수. save_dir 대신 MLflow로 모델을 관리.
    """
    try:
        # 하이퍼파라미터 로드
        hp = HParam(config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            hp_str = ''.join(f.readlines())

        # 체크포인트 설정
        chkpt_path = checkpoint_path if checkpoint_path is not None else None

        # 멀티 프로세싱 설정
        torch.set_num_threads(16)
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"

        import multiprocessing
        multiprocessing.set_start_method('spawn')

        # 데이터 로더 생성
        train_loader, test_loader = create_dataloader(hp, data_dir, data_version)

        # 최신 버전 확인 및 새 버전 설정
        base_dir = "/home/ubuntu/data"
        model_type = "disrupt"
        version = get_latest_version(1) + 1

        # 체크포인트 경로 설정
        save_dir = os.path.join(base_dir, model_type, "model", f"/ver_{version}")
        os.makedirs(save_dir, exist_ok=True)

        chkpt_path = checkpoint_path if checkpoint_path else None

        # MLflow 초기화
        init_mlflow(experiment_name=f"{model_type}_experiment")
        mlflow_run = start_mlflow_run(run_name=f"{model_type}_train")

        # 학습 시작
        train(
            hp=hp,
            train_loader=train_loader,
            valid_loader=valid_loader,
            chkpt_path=chkpt_path,
            save_dir=save_dir,
            db=db,
            version=version,
            data_version=data_version,
        )

    except Exception as e:
        db.rollback()  # 오류 발생 시 롤백
        raise e
