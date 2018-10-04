import dlib

from utils import config_utils

paths = config_utils.get_config("paths")


def get_shape_predictor():
    return dlib.shape_predictor(paths["shapePredictor"])


def get_frontal_face_detector():
    return dlib.get_frontal_face_detector()


def get_face_recog_model():
    return dlib.face_recognition_model_v1(paths["faceRecog1"])
