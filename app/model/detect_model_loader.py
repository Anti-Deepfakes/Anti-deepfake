import torch
import timm
import torch.nn as nn

def load_multimodal_models(weights_path, device):
    model_eye = timm.create_model('inception_resnet_v2', pretrained=False)
    model_nose = timm.create_model('inception_resnet_v2', pretrained=False)
    model_mouth = timm.create_model('inception_resnet_v2', pretrained=False)

    for model in [model_eye, model_nose, model_mouth]:
        model.classif = nn.Sequential(
            nn.Linear(model.classif.in_features, 128),
            nn.ReLU()
        )
    final_layer = nn.Sequential(
        nn.Linear(128 * 3, 1),
        nn.Linear(299 * 3, 299), 
        nn.Sigmoid()
    )


    # 가중치 로드
    checkpoint = torch.load(weights_path, map_location=device)
    model_eye.load_state_dict(checkpoint['model_eye_state_dict'])
    model_nose.load_state_dict(checkpoint['model_nose_state_dict'])
    model_mouth.load_state_dict(checkpoint['model_mouth_state_dict'])
    final_layer.load_state_dict(checkpoint['final_layer_state_dict'])
    
    # 모델과 최종 레이어를 장치로 이동 및 평가 모드 전환
    model_eye.to(device).eval()
    model_nose.to(device).eval()
    model_mouth.to(device).eval()
    final_layer.to(device).eval()

    return model_eye, model_nose, model_mouth, final_layer
