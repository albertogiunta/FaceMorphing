import glob
import os

import dlib
from skimage import io
from skimage.transform import resize


def compile_img_path(img_name=None, img_folder=None):
    img_path = glob.glob(os.path.join("../{}".format(img_folder), img_name))[0]
    print("Processing image: {}".format(img_path))

    return img_path


def load_imgs_in_folder(folder):
    images_paths = glob.glob(folder)
    img_array = []

    for image_path in images_paths:
        loaded_img = load_img_skimage(image_path)
        # loaded_img = load_img_dlib_rgb(image_path)

        # print(numpy.shape(loaded_img))
        # print(loaded_img)

        img_array.append(loaded_img)

    return img_array


def load_img_skimage(img_path):
    from utils import config_utils
    size = config_utils.get_config("preprocessing")["imgSquareSize"]
    return resize(io.imread(img_path), (size, size), anti_aliasing=True, mode="constant")


def load_img_dlib_rgb(img_path):
    return dlib.load_rgb_image(img_path)


def load_img_dlib_grayscale(img_path):
    return dlib.load_grayscale_image(img_path)


def show_img_skimage(image):
    io.imshow(image)
    io.show()


def print_bounding_box(box_index, bounding_box):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(box_index, bounding_box.left(),
                                                                       bounding_box.top(), bounding_box.right(),
                                                                       bounding_box.bottom()))


def show_img_dlib(img, bounding_box=None, shape=None, require_enter=True):
    win = dlib.image_window()
    win.set_image(img)
    if bounding_box is not None:
        win.add_overlay(bounding_box)
    if shape is not None:
        win.add_overlay(shape)
    if require_enter:
        dlib.hit_enter_to_continue()
