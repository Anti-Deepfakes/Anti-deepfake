# 스크립트 실행을 위한 현재 작업 디렉토리 변경
Set-Location -Path "C:\Users\SSAFY\Desktop\project\deep\code\fastapi\S11P31B201\app"

# 현재 작업 디렉토리를 PYTHONPATH로 설정
$env:PYTHONPATH = (Get-Location)

# 호스트와 포트 설정 (변수 이름 변경)
$serverHost = "127.0.0.1"
$serverPort = 8001

# Uvicorn 실행
uvicorn main:app --reload --host $serverHost --port $serverPort