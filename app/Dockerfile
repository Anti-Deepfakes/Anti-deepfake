# Python 3.12 slim 이미지 사용
FROM python:3.12.5-slim

# 빌드 도구와 dlib 설치를 위한 필수 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# dlib 및 Python 패키지 설치
RUN pip install --no-cache-dir dlib

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . /app

# FastAPI 서버 실행
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
