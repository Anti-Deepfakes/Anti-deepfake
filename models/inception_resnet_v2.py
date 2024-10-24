import torch.nn as nn
import timm

def get_model():
    model = timm.create_model('inception_resnet_v2', pretrained=True)
    # Replace the final layer to match the binary classification task (fake or real)
    model.classif = nn.Sequential(
        nn.Linear(model.classif.in_features, 1),
        nn.Sigmoid()
    )
    return model
