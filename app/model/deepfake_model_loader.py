import torch
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


class DeepfakeModel():
    def __init__(self):
        def __load_deepfake_disrupt_model():
            self.__face_detector = FaceAnalysis(name='buffalo_l')
            self.__face_detector.prepare(ctx_id=0, det_size=(224, 224))
            self.__face_swapper = insightface.model_zoo.get_model('./model/disrupt/inswapper_128.onnx', download=False, download_zip=False)
            # print("Model load : ", self.__face_swapper)
        __load_deepfake_disrupt_model()

    def get_deepfake_model(self):
        # print(self.__face_swapper)
        return self.__face_detector, self.__face_swapper
