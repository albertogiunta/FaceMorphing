#   This face self.frontal_face_detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.

import dlib

from utils import config_utils
from utils import img_utils
from utils import models_utils


class Preprocessing:

    def __init__(self):
        self.conf = config_utils.get_config("preprocessing")
        self.img_path = None
        self.img = None
        self.upsample_times = self.conf["upsampleTimes"]
        self.img_square_size = self.conf["imgSquareSize"]
        self.anti_aliasing = bool(self.conf["antiAliasing"])
        self.resize_mode = self.conf["resizeMode"]
        self.frontal_face_detector = models_utils.get_frontal_face_detector()
        self.shape_predictor = models_utils.get_shape_predictor()

    def preprocess_img_path(self, img_path):
        self.img_path = img_path
        self.img = img_utils.load_img_dlib_rgb(img_path)
        return self._preprocess()

    def preprocess_img(self, image):
        self.img_path = "NA"
        self.img = image
        return self._preprocess()

    def _preprocess(self):
        faces_bounding_boxes = self.frontal_face_detector(self.img, self.upsample_times)

        if len(faces_bounding_boxes) < 1:
            print("ERROR: There should be only 1 face per image, instead there are 0 in image {}".format(len(faces_bounding_boxes), self.img_path))
            return None

        faces = dlib.full_object_detections()

        for detection in faces_bounding_boxes:
            faces.append(self.shape_predictor(self.img, detection))
            break

        # img_utils.show_img_dlib(self.img, bounding_box=faces_bounding_boxes, shape=faces[0])

        new_img = dlib.get_face_chip(self.img, faces[0], size=self.img_square_size)
        return new_img


if __name__ == '__main__':
    preproc = Preprocessing()
    img = preproc.preprocess_img_path(
        img_utils.compile_img_path(img_name="00002_930831_fa.png", img_folder="biometix/genuine"))
    img2 = preproc.preprocess_img_path(
        img_utils.compile_img_path(img_name="M_00002_00320.jpg", img_folder="biometix/morphed"))

    img_utils.show_img_skimage(img)
    img_utils.show_img_skimage(img2)
