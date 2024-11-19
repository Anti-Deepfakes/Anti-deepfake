import torch
from model.disrupt.U_Net import UNet


class DisruptModel():
    def __init__(self, weights_path, device):
        def __load_deepfake_disrupt_model():
            pretrained_model = torch.load(weights_path)
            state_dict = {k.replace("module.", ""): v for k, v in pretrained_model['model_state_dict'].items()}

            self.__disrupt_model = UNet(3)
            self.__disrupt_model.load_state_dict(state_dict)
            self.__disrupt_model.to(device).eval()

        __load_deepfake_disrupt_model()


    def get_disrupt_model(self):
        return self.__disrupt_model