import mlflow
import mlflow.pytorch
import os


def init_mlflow(experiment_name: str):
    """
    MLflow 실험 초기화 함수.
    """
    print("[LOG: init_mlflow] Initializing MLflow experiment...")
    print(f"[LOG: init_mlflow] Experiment name: {experiment_name}")
    mlflow.set_tracking_uri("http://mlflow:5000")  # MLflow 서버 URI
    mlflow.set_experiment(experiment_name)  # 실험 이름 설정
    print(f"[LOG: init_mlflow] MLflow initialized with experiment: {experiment_name}")


def start_mlflow_run(run_name: str = None):
    """
    MLflow 실행 시작.
    """
    print("[LOG: start_mlflow_run] Starting MLflow run...")
    print(f"[LOG: start_mlflow_run] Run name: {run_name if run_name else 'Unnamed run'}")
    run = mlflow.start_run(run_name=run_name)
    print(f"[LOG: start_mlflow_run] Started MLflow run with ID: {run.info.run_id}")
    return run


def log_model_with_version(model, model_name: str, base_dir: str, model_type: str, params: dict = None, metrics: dict = None):
    """
    모델 및 메타데이터를 MLflow와 디렉토리 버전 관리 방식으로 저장.
    """
    print("[LOG: log_model_with_version] Logging model with version...")
    print(f"[LOG: log_model_with_version] Model name: {model_name}, Model type: {model_type}")
    print(f"[LOG: log_model_with_version] Base directory: {base_dir}")

    # 현재 디렉토리에서 다음 버전 확인
    version_dir = os.path.join(base_dir, model_type, "model")
    if not os.path.exists(version_dir):
        os.makedirs(version_dir)
        print(f"[LOG: log_model_with_version] Created version directory: {version_dir}")

    current_versions = [int(d.split('_')[1]) for d in os.listdir(version_dir) if d.startswith('ver_')]
    next_version = max(current_versions, default=0) + 1
    version_path = os.path.join(version_dir, f"ver_{next_version}")
    os.makedirs(version_path, exist_ok=True)
    print(f"[LOG: log_model_with_version] Next version: {next_version}, Path: {version_path}")

    # MLflow 저장
    print("[LOG: log_model_with_version] Starting MLflow model logging...")
    with mlflow.start_run():
        if params:
            mlflow.log_params(params)  # 하이퍼파라미터 로깅
            print(f"[LOG: log_model_with_version] Logged parameters: {params}")
        if metrics:
            mlflow.log_metrics(metrics)  # 평가 지표 로깅
            print(f"[LOG: log_model_with_version] Logged metrics: {metrics}")
        mlflow.pytorch.save_model(model, version_path)
        print(f"[LOG: log_model_with_version] Model saved to: {version_path}")

    return next_version


def load_model_from_version(base_dir: str, model_type: str, version: int):
    """
    디렉토리 구조 기반으로 버전에 따라 모델 로드.
    """
    print("[LOG: load_model_from_version] Loading model by version...")
    print(f"[LOG: load_model_from_version] Base directory: {base_dir}, Model type: {model_type}, Version: {version}")
    model_dir = os.path.join(base_dir, model_type, "model", f"ver_{version}")
    model_path = os.path.join(model_dir, f"{model_type}_ver_{version}.pth")

    if not os.path.exists(model_path):
        error_msg = f"Model for version {version} not found in {model_path}."
        print(f"[ERROR: load_model_from_version] {error_msg}")
        raise FileNotFoundError(error_msg)

    print(f"[LOG: load_model_from_version] Model found. Loading from: {model_path}")
    return torch.load(model_path)


def get_latest_version(base_dir: str, model_type: str):
    """
    디렉토리 구조 기반으로 최신 버전 번호를 반환.
    """
    print("[LOG: get_latest_version] Retrieving the latest model version...")
    print(f"[LOG: get_latest_version] Base directory: {base_dir}, Model type: {model_type}")
    model_dir = os.path.join(base_dir, model_type, "model")
    
    existing_versions = [
        int(d.split("_")[-1]) for d in os.listdir(model_dir) if d.startswith("ver_") and d.split("_")[-1].isdigit()
    ]

    if not existing_versions:
        error_msg = f"No versions found for {model_type} in {model_dir}"
        print(f"[ERROR: get_latest_version] {error_msg}")
        raise ValueError(error_msg)

    latest_version = max(existing_versions)
    print(f"[LOG: get_latest_version] Latest version for {model_type}: {latest_version}")
    return latest_version
