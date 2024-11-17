from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.controller.disrupt_controller import router as disrupt_router
from app.config.database import Base, engine
import multiprocessing
import torch
import os

app = FastAPI()

# CORS 설정
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("[LOG: app] CORS middleware configured with origins set to '*'.")

@app.on_event("startup")
def startup_event():
    # 데이터베이스 테이블 생성
    print("[LOG: startup_event] Application startup event triggered.")
    try:
        print("[LOG: startup_event] Creating database tables if not already created.")
        Base.metadata.create_all(bind=engine)
        print("[LOG: startup_event] Database tables created successfully.")
    except Exception as e:
        print(f"[ERROR: startup_event] Error occurred during database table creation: {str(e)}")
        raise

    # 멀티프로세싱 설정
    print("[LOG: setup_multiprocessing] Setting multiprocessing start method.")
    try:
        multiprocessing.set_start_method("spawn", force=True)
        print("[LOG: setup_multiprocessing] Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"[ERROR: setup_multiprocessing] {str(e)}")

    # PyTorch 스레드 및 GPU 설정
    print("[LOG: startup_event] Configuring PyTorch threading and GPU settings.")
    torch.set_num_threads(16)
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    print("[LOG: startup_event] PyTorch configured: Threads=16, CUDA_VISIBLE_DEVICES=7")

    # W&B API 키 가져오기 및 로그인
    print("[LOG: startup_event] Initializing Weights and Biases (W&B).")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print("[LOG: startup_event] W&B login successful.")
    else:
        raise ValueError("[ERROR] W&B API key not found in environment variables.")

# 라우터 추가
app.include_router(disrupt_router, prefix="/disrupt-train")
print("[LOG: app] Router for '/disrupt' added successfully.")

# 글로벌 HTTP 예외 핸들러 추가
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    모든 HTTPException에 대해 JSON 형식의 커스텀 응답 반환
    """
    print(f"[LOG: http_exception_handler] HTTP exception occurred. Path: {request.url.path}, Detail: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.get("/")
def read_root():
    print("[LOG: read_root] Root endpoint accessed. Returning server status message.")
    return {"message": "Disrupt Training Server is Running!"}
