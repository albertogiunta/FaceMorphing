import re

from launcher import genuine
from utils import img_utils


def get_genuine_genuine_pairs(n_images):
    pairs = []
    pattern_extract_number_groups = re.compile('([0-9]{5})')

    for i, img_path in enumerate(genuine):
        current_genuine_id = re.findall(pattern_extract_number_groups, img_path)[0]
        pattern_current_genuine_id = re.compile('\/' + current_genuine_id + 'f')

        next = i + 1
        while pattern_current_genuine_id.search(genuine[next]) is not None:
            pairs.append((img_utils.load_img_dlib_rgb(img_path), img_utils.load_img_dlib_rgb(genuine[next])))
            next += 1

            if len(pairs) >= n_images:
                return pairs

    return pairs


def get_genuine_images(n_images):
    imgs = []
    for i, img_path in enumerate(genuine):
        imgs.append(img_utils.load_img_dlib_rgb(img_path))

        if len(imgs) >= n_images:
            return imgs

    return imgs


def get_nth_genuine_image2(index):
    return _get_nth_image(index, genuine)


def _get_nth_image(index, collection):
    return img_utils.load_img_dlib_rgb(collection[index])

if __name__ == '__main__':
    pairs = get_genuine_genuine_pairs(10)
    for pair in pairs:
        img_utils.show_imgs_skimage(pair)
