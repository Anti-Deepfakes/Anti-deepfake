from fastapi import FastAPI
from app.routers import photo_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# 라우터 등록
app.include_router(photo_router.router ,  prefix="/api")
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)