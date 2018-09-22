#   This face self.frontal_face_detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.

import dlib
from skimage import io

import utils.img_utils as img_utils
import utils.models_utils as models_utils


class Preprocessing:

    def __init__(self, shape_predictor_path):
        self.frontal_face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def preprocess(self, img_path):
        img = img_utils.load_img_dlib(img_path)

        upsample_times = 1

        faces_bounding_boxes = self.frontal_face_detector(img, upsample_times)

        if len(faces_bounding_boxes) != 1:
            print("ERROR: There should be only 1 face per image, instead there are {} in image {}".format(
                len(faces_bounding_boxes), img_path))
            exit()

        # img_utils.show_img_dlib(img, bounding_box=faces_bounding_boxes)

        for box_index, bounding_box in enumerate(faces_bounding_boxes):
            self.__print_bounding_box(box_index, bounding_box)

        faces = dlib.full_object_detections()
        for detection in faces_bounding_boxes:
            faces.append(self.shape_predictor(img, detection))

        img = dlib.get_face_chip(img, faces[0], size=246)
        # img_utils.show_img_dlib(img)
        # img = dlib.get_face_chips(img, faces, size=246)
        # img_utils.show_img_dlib(img[0])

        return img

    @staticmethod
    def __print_bounding_box(box_index, bounding_box):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(box_index, bounding_box.left(),
                                                                           bounding_box.top(), bounding_box.right(),
                                                                           bounding_box.bottom()))


if __name__ == '__main__':
    shape_predictor_path, _, _ = models_utils.model_paths(from_argv=False)
    preproc = Preprocessing(shape_predictor_path)

    img = preproc.preprocess(img_utils.compile_img_path(img_name="stortissimo0.png", img_folder="img"))

    io.imshow(img)
    io.show()
