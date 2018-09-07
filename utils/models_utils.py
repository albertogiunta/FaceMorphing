import sys
import os


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
