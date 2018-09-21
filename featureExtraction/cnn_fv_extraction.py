import dlib
import numpy as np

from featureExtraction.abstract_extraction import AbstractFVExtraction
from utils import models_utils


class CNNFeatureVectorExtraction(AbstractFVExtraction):

    def __init__(self, shape_predictor_path, face_rec_model_path):
        self.frontal_face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

    def get_img_descriptor(self, img_path=None, img_name=None, img_folder=None):
        img = self.load_img_dlib(self.compile_img_path(img_path, img_name, img_folder))

        upsample_times = 1  # Upsampling will make everything bigger and allow us to detect more faces.
        num_jitters = 0  # NB this makes time complexity increase linearly

        faces_bounding_boxes = self.frontal_face_detector(img, upsample_times)

        if len(faces_bounding_boxes) != 1:
            print("ERROR: There should be only 1 face per image, instead there are {} in image {}".format(
                len(faces_bounding_boxes), img_path))
            exit()

        for box_index, bounding_box in enumerate(faces_bounding_boxes):
            shape = self.shape_predictor(img, bounding_box)
            feature_vector = np.array(self.face_rec_model.compute_face_descriptor(img, shape, num_jitters))

            # self.show_img_dlib(img, bounding_box, shape)
            # print(feature_vector)

            return feature_vector

    @staticmethod
    def __print_bounding_box(box_index, bounding_box):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(box_index, bounding_box.left(),
                                                                           bounding_box.top(), bounding_box.right(),
                                                                           bounding_box.bottom()))

    @staticmethod
    def calculate_euclidean_distance(fv_a, fv_b):
        return np.linalg.norm(fv_a - fv_b)

    @staticmethod
    def is_pair_from_same_person(euclidean_distance):
        return euclidean_distance < 0.6


if __name__ == '__main__':
    shape_predictor_path, face_rec_model_path, faces_pairs_path = models_utils.model_paths(from_argv=True)

    face_rec = CNNFeatureVectorExtraction(shape_predictor_path, face_rec_model_path)
    feature_vector = face_rec.get_img_descriptor(img_name="george1.png", img_folder="img")
    print(feature_vector)
