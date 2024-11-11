from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from service.disrupt_service import generate_disrupt_v1
from response.success_response import SuccessResponse

router = APIRouter()

@router.post("/generate")
async def generate_disrupt(request: Request, image: UploadFile = File(...)):
    # print(request.app.state.model_disrupt)
    outputs = await generate_disrupt_v1(request.app.state.model_disrupt, image, request.app.state.device)
    return SuccessResponse.ok(outputs).dict()
