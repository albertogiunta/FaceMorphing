import numpy as np

from featureExtraction.abstract_extraction import AbstractFVExtraction
from utils import config_utils
from utils import img_utils
from utils import models_utils


class CNNFeatureVectorExtraction(AbstractFVExtraction):

    def __init__(self):
        self.conf = config_utils.get_config("FVExtractionCnn")
        self.img = None
        self.frontal_face_detector = models_utils.get_frontal_face_detector()
        self.shape_predictor = models_utils.get_shape_predictor()
        self.face_rec_model = models_utils.get_face_recog_model()
        self.upsample_times = self.conf["upsampleTimes"]
        self.num_jitters = self.conf["numJitters"]

    def get_img_descriptor_from_img(self, img):
        self.img = img
        return self._get_img_descriptor()

    def get_img_descriptor_from_path(self, img_path):
        self.img = img_utils.load_img_dlib_rgb(img_path)
        return self._get_img_descriptor()

    def _get_img_descriptor(self):
        faces_bounding_boxes = self.frontal_face_detector(self.img, self.upsample_times)

        for box_index, bounding_box in enumerate(faces_bounding_boxes):
            shape = self.shape_predictor(self.img, bounding_box)
            # img_utils.show_img_dlib(self.img, bounding_box, shape)
            feature_vector = np.array(self.face_rec_model.compute_face_descriptor(self.img, shape, self.num_jitters))
            return feature_vector


if __name__ == '__main__':
    face_rec = CNNFeatureVectorExtraction()
    feature_vector = face_rec.get_img_descriptor_from_path(
        img_utils.compile_img_path(img_name="00002_930831_fa.png", img_folder="biometrix/genuine"))
    print(feature_vector)

    # feature_vector2 = face_rec.get_img_descriptor_from_path(
    #     img_utils.compile_img_path(img_name="M_00002_00320.jpg", img_folder="biometrix/morphed"))
    # print(feature_vector2)
