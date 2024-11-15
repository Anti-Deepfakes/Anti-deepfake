from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from controller.disrupt_controller import router as disrupt_router
from app.config.database import Base, engine

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

@app.on_event("startup")
def startup_event():
    # 데이터베이스 테이블 생성
    Base.metadata.create_all(bind=engine)

# 라우터 추가
app.include_router(disrupt_router, prefix="/disrupt")

# 글로벌 HTTP 예외 핸들러 추가
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    모든 HTTPException에 대해 JSON 형식의 커스텀 응답 반환
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.get("/")
def read_root():
    return {"message": "Disrupt Training Server is Running!"}
