import re

import numpy

from featureVectorMerging.differential_comparison import DifferentialComparison
from utils import img_utils


def get_morphed_images(morphed):
    imgs = []
    for img in morphed:
        imgs.append((img_utils.load_img_dlib_rgb(img), img_basename(img)))
    return imgs[:500]


def get_genuine_images(genuine):
    imgs = []
    for img in genuine:
        imgs.append((img_utils.load_img_dlib_rgb(img), img_basename(img)))
    return imgs[:500]


def get_morphed_labels(morphed):
    lbls = []
    for img in morphed:
        lbls.append(img_basename(img))
    return lbls[:500]


def get_genuine_labels(genuine):
    lbls = []
    for img in genuine:
        lbls.append(img_basename(img))
    return lbls[:500]


def img_basename(name):
    try:
        str = re.compile(r'(?:imgs\/)(.*)(?:\.*)').findall(name)[0]
        return str
    except IndexError:
        print(name)


def get_morphed_genuine_pairs(morphed, genuine, db_name, method):
    DB_MORPHEDDB = "morphedDB"
    DB_PMDB_DIG = "pmDB"
    DB_BIOMETIX = "biometix"
    pairs = []

    if method == "DFC_CNN_OF":
        for el in morphed:
            path = el[0]
            fv = el[1]
            if db_name == DB_MORPHEDDB:
                genuine_id = list(re.compile(r'-([0-9]{5})_|([0-9]{5})d').findall(path)[1])
            elif db_name == DB_PMDB_DIG:
                genuine_id = list(re.compile(r'(?:__)([a-z]-[0-9]{3})-[0-9]{1,}(?:__[0-9])|(?:__)([0-9]{5})_[0-9]{6}_[a-z]{2}(?:__[0-9])|([0-9]{5})d[0-9]{2,}(?:__[0-9])').findall(path)[0])

            while '' in genuine_id: genuine_id.remove('')
            genuine_id = genuine_id[0]
            try:
                genuine_fv = next(x[1] for x in genuine if re.compile('(.+' + genuine_id + '.+)').search(x[0]) is not None)
            except StopIteration:
                print("While getting morphed-genuine pairs i've not found image " + genuine_id + " " + path)
                continue
            pairs.append((DifferentialComparison.get_differential_fv(numpy.array(fv), numpy.array(genuine_fv)), img_basename(path) + "-" + genuine_id))
    else:
        for i, img in enumerate(morphed):
            if db_name == DB_MORPHEDDB:
                genuine_id = list(re.compile(r'-([0-9]{5})_|([0-9]{5})d').findall(img)[1])
            elif db_name == DB_PMDB_DIG:
                genuine_id = list(re.compile(r'(?:__)([a-z]-[0-9]{3})-[0-9]{1,}(?:__[0-9])|(?:__)([0-9]{5})_[0-9]{6}_[a-z]{2}(?:__[0-9])|([0-9]{5})d[0-9]{2,}(?:__[0-9])').findall(img)[0])
            elif db_name == DB_BIOMETIX:
                genuine_id = list(re.compile(r'(?:._)([0-9]{5})').findall(img)[0])

            while '' in genuine_id: genuine_id.remove('')
            genuine_id = genuine_id[0]
            try:
                genuine_img = next(x for x in genuine if re.compile('(.+' + genuine_id + '.+)').search(x) is not None)
            except StopIteration:
                print("While getting morphed-genuine pairs i've not found image " + genuine_id + " " + img)
                continue
            if len(pairs) > 500: return pairs
            pairs.append((img_utils.load_img_dlib_rgb(img), img_utils.load_img_dlib_rgb(genuine_img), img_basename(img) + "-" + genuine_id))
    return pairs


def get_genuine_genuine_pairs(genuine, genuine4morphed, img_type, db_name, method):
    DB_MORPHEDDB = "morphedDB"
    DB_PMDB_DIG = "pmDB"
    DB_BIOMETIX = "biometix"

    pairs = []

    if method != "DFC_CNN_OF":
        genuine4morphed = sorted(genuine4morphed)
        genuine = sorted(genuine)

    for i, img in enumerate(genuine):
        if db_name == DB_MORPHEDDB or db_name == DB_BIOMETIX:
            if method == "DFC_CNN_OF":
                current_genuine_id = re.compile('([0-9]{5})').findall(img[0])[0]
            else:
                current_genuine_id = re.compile('([0-9]{5})').findall(img)[0]

            pattern_current_genuine_id = re.compile(current_genuine_id + '[d|_|f]')
            if img_type == "digital":
                next_img = i + 1
                if method == "DFC_CNN_OF":
                    path = img[0]
                    fv = img[1]
                    while pattern_current_genuine_id.search(genuine[next_img][0]) is not None:
                        if len(pairs) >= len(genuine): return pairs
                        pairs.append((DifferentialComparison.get_differential_fv(numpy.array(fv), numpy.array(genuine[next_img][1])), img_basename(path) + "-" + img_basename(genuine[next_img][0])))
                        next_img += 1
                else:
                    while pattern_current_genuine_id.search(genuine[next_img]) is not None:
                        if len(pairs) >= len(genuine) or len(pairs) > 500: return pairs
                        pairs.append((img_utils.load_img_dlib_rgb(img), img_utils.load_img_dlib_rgb(genuine[next_img]), img_basename(img) + "-" + img_basename(genuine[next_img])))
                        next_img += 1
            elif img_type == "ps":
                if method == "DFC_CNN_OF":
                    path = img[0]
                    fv = img[1]
                    try:
                        genuine_img = next(x for x in genuine4morphed if re.compile('(.+' + current_genuine_id + '.+)').search(x[0]) is not None)
                    except StopIteration:
                        print("While getting morphed-genuine pairs i've not found image " + current_genuine_id + " " + img)
                        continue
                    pairs.append((DifferentialComparison.get_differential_fv(numpy.array(fv), numpy.array(genuine_img[1])), img_basename(path) + "-" + img_basename(genuine_img[0])))
                else:
                    try:
                        genuine_img = next(x for x in genuine4morphed if re.compile('(.+' + current_genuine_id + '.+)').search(x) is not None)
                    except StopIteration:
                        print("While getting morphed-genuine pairs i've not found image " + current_genuine_id + " " + img)
                        continue
                    pairs.append((img_utils.load_img_dlib_rgb(img), img_utils.load_img_dlib_rgb(genuine_img), img_basename(img) + "-" + img_basename(genuine_img)))
        elif db_name == DB_PMDB_DIG:
            if method == "DFC_CNN_OF":
                path = img[0]
                fv = img[1]
                pairs.append((DifferentialComparison.get_differential_fv(numpy.array(fv), numpy.array(genuine4morphed[i][1])), img_basename(path) + "-" + img_basename(genuine4morphed[i][0])))
            else:
                pairs.append((img_utils.load_img_dlib_rgb(img), img_utils.load_img_dlib_rgb(genuine4morphed[i]), img_basename(img) + "-" + img_basename(genuine4morphed[i])))

    return pairs

if __name__ == '__main__':
    # print(get_genuine_genuine_diff_fv())
    # for el in get_genuine_genuine_pairs():
    #     if len(el) != 3:
    #         print(el)
    pass
