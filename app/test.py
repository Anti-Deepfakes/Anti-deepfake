import torch

checkpoint = torch.load("model/detect/best.pth", map_location="cpu")
print(checkpoint.keys())  # 체크포인트 파일의 모든 키 출력
