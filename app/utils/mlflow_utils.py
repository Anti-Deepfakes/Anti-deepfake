import mlflow
import mlflow.pytorch
import os


def init_mlflow(experiment_name: str):
    """
    MLflow 실험 초기화 함수.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")  # MLflow 서버 URI
    mlflow.set_experiment(experiment_name)  # 실험 이름 설정
    print(f"MLflow initialized with experiment: {experiment_name}")

def start_mlflow_run(run_name: str = None):
    """
    MLflow 실행 시작.
    """
    run = mlflow.start_run(run_name=run_name)
    print(f"Started MLflow run: {run.info.run_id}")
    return run

def log_model_with_version(model, model_name: str, base_dir: str, model_type: str, params: dict = None, metrics: dict = None):
    """
    모델 및 메타데이터를 MLflow와 디렉토리 버전 관리 방식으로 저장.
    """
    # 현재 디렉토리에서 다음 버전 확인
    version_dir = os.path.join(base_dir, model_type, "model")
    if not os.path.exists(version_dir):
        os.makedirs(version_dir)

    current_versions = [int(d.split('_')[1]) for d in os.listdir(version_dir) if d.startswith('ver_')]
    next_version = max(current_versions, default=0) + 1
    version_path = os.path.join(version_dir, f"ver_{next_version}")
    os.makedirs(version_path, exist_ok=True)

    # MLflow 저장
    with mlflow.start_run():
        if params:
            mlflow.log_params(params)  # 하이퍼파라미터 로깅
        if metrics:
            mlflow.log_metrics(metrics)  # 평가 지표 로깅
        mlflow.pytorch.save_model(model, version_path)
        print(f"Model saved to {version_path} (version {next_version})")

    return next_version


def load_model_from_version(base_dir: str, model_type: str, version: int):
    """
    디렉토리 구조 기반으로 버전에 따라 모델 로드.
    Args:
        base_dir: 모델 저장 루트 디렉토리.
        model_type: 모델 종류 ("disrupt" 또는 "detect").
        version: 로드할 버전 번호.
    """
    model_dir = os.path.join(base_dir, model_type, "model", f"ver_{version}")
    model_path = os.path.join(model_dir, f"{model_type}_ver_{version}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for version {version} not found in {model_path}.")

    print(f"Loading model from: {model_path}")
    return torch.load(model_path)


def get_latest_version(base_dir: str, model_type: str):
    """
    디렉토리 구조 기반으로 최신 버전 번호를 반환.
    Args:
        base_dir: 모델 저장 루트 디렉토리.
        model_type: 모델 종류 ("disrupt" 또는 "detect").
    """
    model_dir = os.path.join(base_dir, model_type, "model")
    existing_versions = [
        int(d.split("_")[-1]) for d in os.listdir(model_dir) if d.startswith("ver_") and d.split("_")[-1].isdigit()
    ]

    if not existing_versions:
        raise ValueError(f"No versions found for {model_type} in {model_dir}")

    latest_version = max(existing_versions)
    print(f"Latest version for {model_type}: {latest_version}")
    return latest_version