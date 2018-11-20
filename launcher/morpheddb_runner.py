import re

import numpy

from featureVectorMerging.differential_comparison import DifferentialComparison
from launcher import morphed, genuine, genuine4morphed, sorted_morphed_labels_fvs, sorted_genuine4morphed_labels_fvs, sorted_genuine_labels_fvs, MAX_GENUINES
from utils import img_utils


def get_morphed_images():
    imgs = []
    for img in morphed:
        imgs.append((img_utils.load_img_dlib_rgb(img), img_basename(img)))
    return imgs

def get_genuine_images():
    imgs = []
    for img in genuine:
        imgs.append((img_utils.load_img_dlib_rgb(img), img_basename(img)))
    return imgs


def get_morphed_labels():
    lbls = []
    for img in morphed:
        lbls.append(img_basename(img))
    return lbls


def get_genuine_labels():
    lbls = []
    for img in genuine:
        lbls.append(img_basename(img))
    return lbls


def img_basename(name):
    return re.compile(r'(?:imgs\/)(.*)(?:\.png)').findall(name)[0]

def get_morphed_genuine_pairs():
    pairs = []
    for img in morphed:
        genuine_id = list(re.compile(r'([0-9d]{8,})|([0-9_]{12})').findall(img)[1])
        genuine_id.remove('')
        genuine_id = genuine_id[0]
        try:
            genuine = next(x for x in genuine4morphed if re.compile('(.+' + genuine_id + '.+)').search(x) is not None)
        except StopIteration:
            print("While getting morphed-genuine pairs i've not found image " + genuine_id)
            continue
        pairs.append((img_utils.load_img_dlib_rgb(img), img_utils.load_img_dlib_rgb(genuine), img_basename(img) + "-" + genuine_id))
    return pairs


def get_genuine_genuine_pairs():
    pairs = []

    for i, img in enumerate(sorted(genuine)):
        current_genuine_id = re.compile('([0-9]{5})').findall(img)[0]
        pattern_current_genuine_id = re.compile(current_genuine_id + '[d|_]')
        next = i + 1
        while pattern_current_genuine_id.search(genuine[next]) is not None:
            if len(pairs) == MAX_GENUINES:
                return pairs
            pairs.append((img_utils.load_img_dlib_rgb(img), img_utils.load_img_dlib_rgb(genuine[next]), img_basename(img) + "-" + img_basename(genuine[next])))
            next += 1
    return pairs


def get_morphed_genuine_diff_fv():
    pairs = []
    for el in sorted_morphed_labels_fvs:
        path = el[0]
        fv = el[1]
        genuine_id = list(re.compile(r'([0-9d]{8,})|([0-9_]{12})').findall(path)[1])
        genuine_id.remove('')
        genuine_id = genuine_id[0]
        pattern_current_genuine_id = re.compile('(.+' + genuine_id + '.+)')
        genuine_fv = next(x[1] for x in sorted_genuine4morphed_labels_fvs if pattern_current_genuine_id.search(x[0]) is not None)
        pairs.append((DifferentialComparison.get_differential_fv(numpy.array(fv), numpy.array(genuine_fv)), img_basename(path) + "-" + genuine_id))
    return pairs


def get_genuine_genuine_diff_fv():
    pairs = []
    for i, el in enumerate(sorted_genuine_labels_fvs):
        path = el[0]
        fv = el[1]
        current_genuine_id = re.compile('([0-9]{5})').findall(path)[0]
        pattern_current_genuine_id = re.compile(current_genuine_id + '[d|_]')
        next_i = i + 1
        while pattern_current_genuine_id.search(sorted_genuine_labels_fvs[next_i][0]) is not None:
            if len(pairs) >= MAX_GENUINES:
                return pairs
            pairs.append((DifferentialComparison.get_differential_fv(numpy.array(fv), numpy.array(sorted_genuine_labels_fvs[next_i][1])),
                          img_basename(path) + "-" + img_basename(sorted_genuine_labels_fvs[next_i][0])))
            next_i += 1
    return pairs


if __name__ == '__main__':
    # print(get_genuine_genuine_diff_fv())
    # for el in get_genuine_genuine_pairs():
    #     if len(el) != 3:
    #         print(el)
    pass
