#FROM python:3.12.5-slim
FROM nvcr.io/nvidia/pytorch:24.02-py3

RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir dlib

# 기본 작업 디렉토리 설정
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install opencv-fixer==0.2.5 && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]