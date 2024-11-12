from fastapi import FastAPI, HTTPException, Request
from controller.detect_controller import router as image_router
from fastapi.responses import JSONResponse
from model.detect_model_loader import load_multimodal_models
from controller.trigger_controller import trigger_router
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup_event():
    # 모델을 FastAPI의 app.state에 저장
    app.state.model_eye, app.state.model_nose, app.state.model_mouth, app.state.final_layer = load_multimodal_models("model/detect/model_checkpoint_epoch_24.pth", device)
    app.state.device = device

app.include_router(image_router, prefix="/api/detect")
app.include_router(trigger_router, prefix="/model")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
