#!/bin/bash

# 환경 변수 설정 (필요한 경우 수정)
export PYTHONPATH=$(pwd)

# 포트 설정 (필요에 따라 수정 가능)
HOST="0.0.0.0"
PORT="8000"

# Uvicorn 실행
uvicorn app.main:app --reload --host $HOST --port $PORT