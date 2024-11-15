# 1. 베이스 이미지를 Python 3.12로 설정
FROM python:3.12-slim

# 2. 작업 디렉토리를 설정
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential

# 3. 시스템 의존성 설치 (여기서는 OpenCV와 같은 패키지를 설치)
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 필요 라이브러리 설치
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 5. FastAPI 서버의 소스 코드를 컨테이너에 복사
COPY . .

# 6. FastAPI를 실행하기 위한 커맨드 설정
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

