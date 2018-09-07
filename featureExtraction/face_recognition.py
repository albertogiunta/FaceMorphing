import glob

import dlib
import numpy as np

from utils import models_utils


class FaceRecognition:
    def __init__(self, shape_predictor_path, face_rec_model_path):
        self.frontal_face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

    def get_descriptors_for_images(self, faces_pair_path):
        descriptors = []

        images = glob.glob(faces_pair_path)

        if len(images) > 2:
            print("ERROR: There are more than 2 pictures in the specified folder ({})".format(faces_pairs_path))
            exit(0)

        for img in images:
            print("Processing image: {}".format(img))
            descriptors.append(self.get_img_descriptor(img))

        return np.array(descriptors)

    def get_img_descriptor(self, img):
        upsample_times = 1  # Upsampling will make everything bigger and allow us to detect more faces.
        num_jitters = 0  # NB this makes time complexity increase linearly
        img = dlib.load_rgb_image(img)

        faces_bounding_boxes = self.frontal_face_detector(img, upsample_times)

        for box_index, bounding_box in enumerate(faces_bounding_boxes):
            shape = self.shape_predictor(img, bounding_box)
            face_descriptor = np.array(self.face_rec_model.compute_face_descriptor(img, shape, num_jitters))

            # self.__display_image(img, bounding_box, shape)
            # print(face_descriptor)

            return face_descriptor

    @staticmethod
    def __print_bounding_box(box_index, bounding_box):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(box_index, bounding_box.left(),
                                                                           bounding_box.top(), bounding_box.right(),
                                                                           bounding_box.bottom()))

    @staticmethod
    def __display_image(img, bounding_box, shape, require_enter=True):
        win = dlib.image_window()
        win.set_image(img)
        win.add_overlay(bounding_box)
        win.add_overlay(shape)
        if require_enter:
            dlib.hit_enter_to_continue()

    @staticmethod
    def calculate_euclidean_distance(fv_a, fv_b):
        return np.linalg.norm(fv_a - fv_b)

    @staticmethod
    def is_pair_from_same_person(euclidean_distance):
        return euclidean_distance < 0.6


if __name__ == '__main__':
    shape_predictor_path, face_rec_model_path, faces_pairs_path = models_utils.model_paths(from_argv=True)

    face_rec = FaceRecognition(shape_predictor_path, face_rec_model_path)
    feature_vectors = face_rec.get_descriptors_for_images(faces_pairs_path)
    euclidean_distance = face_rec.calculate_euclidean_distance(feature_vectors[0], feature_vectors[1])

    print("Match found: {}".format(face_rec.is_pair_from_same_person(euclidean_distance)))
    print("Euclidean distance: {}".format(euclidean_distance))
