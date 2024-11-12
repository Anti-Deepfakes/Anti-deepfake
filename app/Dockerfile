# Python 3.12 slim 이미지 사용
FROM python:3.12.5-slim

# 빌드 도구와 dlib 설치를 위한 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# dlib 및 다른 Python 패키지 설치
RUN pip install --no-cache-dir dlib

# 모델 파일을 Docker 이미지의 /app/models 폴더로 복사
RUN mkdir -p /app/models
COPY ./models/shape_predictor_68_face_landmarks.dat /app/models/

# 나머지 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사 및 작업 디렉토리 설정
COPY . /app
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
