from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from service.deepfake_service import generate_deepfake_v1
from response.success_response import SuccessResponse

router = APIRouter()

@router.post("/generate")
async def generate_deepfake(request: Request, image: UploadFile = File(...)):
    outputs = await generate_deepfake_v1(request.app.state.model_face_detector, request.app.state.model_deepfake, image)
    return SuccessResponse.ok(outputs).dict()
