from fastapi import FastAPI
from app.routers import photo_router

app = FastAPI()

# 라우터 등록
app.include_router(photo_router.router ,  prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)