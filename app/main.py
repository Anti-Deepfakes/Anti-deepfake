from fastapi import FastAPI, HTTPException, Request
from controller.disrupt_controller import router as disrupt_router
from controller.deepfake_controller import router as deepfake_router
from fastapi.responses import JSONResponse
from model.disrupt_model_loader import DisruptModel
from model.deepfake_model_loader import DeepfakeModel
# from controller.trigger_controller import trigger_router
import os
import torch
from fastapi.middleware.cors import CORSMiddleware

disrupt_model = os.getenv("DISRUPT_MODEL") # "/home/ubuntu/model/disrupt/unet_epoch_2.pth"
deepfake_model = os.getenv("DEEPFAKE_MODEL") # "/home/ubuntu/model/disrupt/inswapper_128.onnx"

app = FastAPI()
origins = [
	"*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup_event():
    # 모델을 FastAPI의 app.state에 저장
    app.state.model_disrupt = DisruptModel(disrupt_model, device).get_disrupt_model()
    app.state.model_face_detector, app.state.model_deepfake = DeepfakeModel(deepfake_model).get_deepfake_model()
    # print(app.state.model_deepfake)
    app.state.device = device

app.include_router(disrupt_router, prefix="/disrupt/disrupt")
app.include_router(deepfake_router, prefix="/disrupt/deepfake")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.get("/disrupt")
def read_root():
    return {"Hello": "World"}
