import os
import sys

import dlib

from utils import config_utils

paths = config_utils.get_config("paths")


def model_paths(from_argv=False, img_format="png"):
    if from_argv:
        predictor_path = sys.argv[1]
        face_rec_model_path = sys.argv[2]
        faces_folder_path = sys.argv[3]
    else:
        predictor_path = "../models/shape_predictor_5_face_landmarks.dat"
        face_rec_model_path = "../models/dlib_face_recognition_resnet_model_v1.dat"
        faces_folder_path = "../img"

    return predictor_path, face_rec_model_path, os.path.join(faces_folder_path, "*." + img_format)


def get_shape_predictor():
    return dlib.shape_predictor(paths["shapePredictor"])


def get_frontal_face_detector():
    return dlib.get_frontal_face_detector()


def get_face_recog_model():
    return dlib.face_recognition_model_v1(paths["faceRecog1"])
