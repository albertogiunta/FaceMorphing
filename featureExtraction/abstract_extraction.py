import glob
import os
from abc import abstractmethod

import dlib
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize


class AbstractFVExtraction:

    def get_img_pair_descriptors(self, faces_pair_folder_path):
        descriptors = []

        images = glob.glob(faces_pair_folder_path)

        if len(images) > 2:
            print("ERROR: There are more than 2 pictures in the specified folder ({})".format(faces_pair_folder_path))
            exit(0)

        for img_path in images:
            descriptors.append(self.get_img_descriptor(img_path=img_path))

        return np.array(descriptors)

    @abstractmethod
    def get_img_descriptor(self, img_path=None, img_name=None, img_folder=None):
        pass

    def compile_img_path(self, img_path=None, img_name=None, img_folder=None):
        if img_path is None:
            img_path = glob.glob(os.path.join("../{}".format(img_folder), img_name))[0]
        print("Processing image: {}".format(img_path))

        return img_path

    @staticmethod
    def load_img_skimage(img_path):
        return resize(rgb2gray(io.imread(img_path)), (256, 256), anti_aliasing=True, mode="constant")

    @staticmethod
    def load_img_dlib(img_path):
        return dlib.load_rgb_image(img_path)

    @staticmethod
    def show_img_skimage(image):
        io.imshow(image)
        io.show()

    @staticmethod
    def show_img_dlib(img, bounding_box, shape, require_enter=True):
        win = dlib.image_window()
        win.set_image(img)
        win.add_overlay(bounding_box)
        win.add_overlay(shape)
        if require_enter:
            dlib.hit_enter_to_continue()
