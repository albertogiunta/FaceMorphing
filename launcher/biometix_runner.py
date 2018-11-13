import re

from launcher import morphed, genuine4morphed
from utils import img_utils

pattern_extract_genuine_ids_from_morphed = re.compile(r'([0-9]{5})')


def get_morphed_genuine_pairs(n_images):
    pairs = []

    for img_path in morphed[0:n_images]:
        genuine_id = re.findall(pattern_extract_genuine_ids_from_morphed, img_path)[0]
        pattern_current_genuine_id = re.compile('(.+' + genuine_id + '.+)')
        genuine_path = next(x for x in genuine4morphed if pattern_current_genuine_id.search(x) is not None)
        pairs.append((img_utils.load_img_dlib_rgb(img_path), img_utils.load_img_dlib_rgb(genuine_path)))
    return pairs


def get_morphed_images(n_images):
    return _get_n_images(n_images, morphed)


def get_genuine_images(n_images):
    return _get_n_images(n_images, genuine4morphed)


def get_nth_morphed_image(index):
    return _get_nth_image(index, morphed)


def get_nth_morphed_genuine_pair(index):
    morphed_path = morphed[index]
    genuine_id = re.findall(pattern_extract_genuine_ids_from_morphed, morphed_path)[0]
    pattern_current_genuine_id = re.compile('(.+' + genuine_id + '.+)')
    genuine_path = next(x for x in genuine4morphed if pattern_current_genuine_id.search(x) is not None)
    print(morphed_path)
    print(genuine_path)
    return morphed_path, genuine_path


def get_nth_genuine_image(index):
    return _get_nth_image(index, genuine4morphed)


def _get_n_images(n_images, collection):
    imgs = []

    for img in collection[0:n_images]:
        imgs.append(img_utils.load_img_dlib_rgb(img))

    return imgs


def _get_nth_image(index, collection):
    return img_utils.load_img_dlib_rgb(collection[index])
