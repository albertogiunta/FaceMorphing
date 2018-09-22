import glob
import os

import dlib
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize


def compile_img_path(img_name=None, img_folder=None):
    img_path = glob.glob(os.path.join("../{}".format(img_folder), img_name))[0]
    print("Processing image: {}".format(img_path))

    return img_path


def load_img_skimage(img_path):
    return resize(rgb2gray(io.imread(img_path)), (256, 256), anti_aliasing=True, mode="constant")


def load_img_dlib(img_path):
    return dlib.load_rgb_image(img_path)


def show_img_skimage(image):
    io.imshow(image)
    io.show()


def show_img_dlib(img, bounding_box=None, shape=None, require_enter=True):
    win = dlib.image_window()
    win.set_image(img)
    if bounding_box is not None:
        win.add_overlay(bounding_box)
    if shape is not None:
        win.add_overlay(shape)
    if require_enter:
        dlib.hit_enter_to_continue()
