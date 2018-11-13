import re

import numpy

from featureVectorMerging.differential_comparison import DifferentialComparison
from launcher import morphed, genuine, genuine4morphed, sorted_morphed_labels_fvs, sorted_genuine4morphed_labels_fvs, sorted_genuine_labels_fvs
from utils import img_utils


def get_morphed_images():
    imgs = []
    for img in morphed:
        imgs.append(img_utils.load_img_dlib_rgb(img))
    return imgs


def get_genuine_images():
    imgs = []
    for img in genuine:
        imgs.append(img_utils.load_img_dlib_rgb(img))
    return imgs


def get_morphed_genuine_pairs():
    pairs = []
    for img in morphed:
        genuine_id = list(re.compile(r'([0-9d]{8,})|([0-9_]{12})').findall(img)[1])
        genuine_id.remove('')
        genuine_id = genuine_id[0]
        genuine = next(x for x in genuine4morphed if re.compile('(.+' + genuine_id + '.+)').search(x) is not None)
        pairs.append((img_utils.load_img_dlib_rgb(img), img_utils.load_img_dlib_rgb(genuine)))
    return pairs


def get_genuine_genuine_pairs():
    pairs = []

    for i, img in enumerate(sorted(genuine)):
        current_genuine_id = re.compile('([0-9]{5})').findall(img)[0]
        pattern_current_genuine_id = re.compile(current_genuine_id + '[d|_]')
        next = i + 1
        while pattern_current_genuine_id.search(genuine[next]) is not None:
            pairs.append((img_utils.load_img_dlib_rgb(img), img_utils.load_img_dlib_rgb(genuine[next])))
            next += 1
            if len(pairs) >= 400:
                return pairs
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
        pairs.append(DifferentialComparison.get_differential_fv(numpy.array(fv), numpy.array(genuine_fv)))
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
            pairs.append(DifferentialComparison.get_differential_fv(numpy.array(fv), numpy.array(sorted_genuine_labels_fvs[next_i][1])))
            next_i += 1
            if len(pairs) >= 400:
                return pairs
    return pairs
