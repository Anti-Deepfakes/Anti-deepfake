from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from service.detect_service import ai_inference
from response.detect_success_response import SuccessResponse

router = APIRouter()

@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # app.state에서 모델과 device를 가져옵니다.
        model_eye = request.app.state.model_eye
        model_nose = request.app.state.model_nose
        model_mouth = request.app.state.model_mouth
        final_layer = request.app.state.final_layer
        device = request.app.state.device

        outputs = await ai_inference((model_eye, model_nose, model_mouth, final_layer), file, device)
        return SuccessResponse.ok(outputs).dict()
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="예측 과정에서 오류가 발생했습니다.")
