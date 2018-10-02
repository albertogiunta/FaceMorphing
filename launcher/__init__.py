import glob
import json
import re

import numpy as np
import skimage
from numpy import shape
from skimage import io
from skimage.color import rgb2gray
from skimage.util import view_as_blocks

from SVM.svm_classification import SVMClassifier
from featureExtraction import CNNFeatureVectorExtraction
from featureExtraction.lbph_fv_extraction import LBPHFeatureVectorExtraction
from featureVectorMerging.differential_comparison import DifferentialComparison
from preprocessing.preprocessing import Preprocessing


def get_path_of_morphed_and_base_genuine(morphed_path):
    genuine_ids = re.findall(pattern_extract_genuine_ids_from_morphed, morphed_path)
    base_genuine_id = genuine_ids[1]  # change to 0 to check the other image used for the attack
    pattern_match_genuine = re.compile('(.+' + base_genuine_id + '.+)')
    genuine_path = next(x for x in all_genuine if pattern_match_genuine.search(x) is not None)

    print("Genuine (Base) : {}\n"
          "Morphed: {} + {}\n"
          .format(base_genuine_id, genuine_ids[0], genuine_ids[1]))

    return genuine_path, morphed_path, base_genuine_id, genuine_ids


def apply_preproc(genuine_path, morphed_path):
    preproc_imgs = []
    preproc = Preprocessing()

    preproc_imgs.append(preproc.preprocess_img_path(genuine_path))
    preproc_imgs.append(preproc.preprocess_img_path(morphed_path))

    return preproc_imgs


def apply_cnn(genuine, morphed):
    feature_vectors = []

    genuine = skimage.img_as_ubyte(genuine)
    morphed = skimage.img_as_ubyte(morphed)

    cnn = CNNFeatureVectorExtraction()
    feature_vectors.append(cnn.get_img_descriptor_from_img(genuine))
    feature_vectors.append(cnn.get_img_descriptor_from_img(morphed))

    return feature_vectors


def apply_lbph(genuine, morphed, is_PPC):
    feature_vectors = []

    genuine = rgb2gray(genuine)
    morphed = rgb2gray(morphed)

    lbph = LBPHFeatureVectorExtraction()
    feature_vectors.append(lbph.get_img_descriptor_from_img(genuine, is_PPC))
    feature_vectors.append(lbph.get_img_descriptor_from_img(morphed, is_PPC))
    return feature_vectors


def classify(differential_fv):
    svm = SVMClassifier()
    value = svm.predict_class(differential_fv)
    return value


# image_comparison_method: full_image OR patch_wise
# fv_extraction_method: cnn OR lbph
def get_element_to_be_classified(image_comparison_method, fv_extraction_method, genuine, morphed):
    feature_vectors = None

    if image_comparison_method == "full_image":
        if fv_extraction_method == "cnn":
            feature_vectors = apply_cnn(genuine, morphed)
        elif fv_extraction_method == "lbph":
            feature_vectors = apply_lbph(genuine, morphed, is_PPC=False)
        differential = DifferentialComparison.get_differential_fv(feature_vectors[1], feature_vectors[0])
        return differential

    elif image_comparison_method == "patch_wise":
        patch_size = 16
        patch_shape = (patch_size, patch_size)
        print(shape(genuine))
        patched_genuine = view_as_blocks(rgb2gray(genuine), patch_shape)
        patched_morphed = view_as_blocks(rgb2gray(morphed), patch_shape)
        print(shape(patched_genuine))

        concatenated = []

        for i, row in enumerate(patched_genuine):
            print(i)
            for j, col in enumerate(patched_genuine[i]):
                print("              " + str(j))
                genuine_patch = patched_genuine[i, j]
                morphed_patch = patched_morphed[i, j]
                if fv_extraction_method == "cnn":
                    feature_vectors = apply_cnn(genuine_patch, morphed_patch)
                elif fv_extraction_method == "lbph":
                    feature_vectors = apply_lbph(genuine_patch, morphed_patch, is_PPC=True)

                euclidean = DifferentialComparison.calculate_euclidean_distance(feature_vectors[0], feature_vectors[1])
                concatenated.append(euclidean)

        return concatenated


def show_imgs(imgs):
    io.imshow_collection(imgs)
    io.show()


def dispatch(method):
    data = []
    for i, img in enumerate(all_morphed):
        if imgs_to_be_checked != 0 and i >= imgs_to_be_checked:
            break

        print("Processign mage #{}".format(i))
        genuine_path, morphed_path, genuine_id, (morphed_id_1, morphed_id_2) = get_path_of_morphed_and_base_genuine(img)

        # preprocess: 1st genuine, 2nd morphed
        preproc_imgs = apply_preproc(genuine_path, morphed_path)
        # show_imgs(preproc_imgs)

        full_image_method = "full_image"
        patch_wise_method = "patch_wise"
        lbph_method = "lbph"
        cnn_method = "cnn"

        image_data = {
            "investigated_image_name": morphed_id_1 + "_" + morphed_id_2,
            "probe_image_name": genuine_id,
            "class": -1,
            "classification_data": []
        }

        if method == "FVC_CNN":
            print("todo")
        elif method == "DFC_CNN":
            image_data["classification_data"] = get_element_to_be_classified(full_image_method, cnn_method,
                                                                             preproc_imgs[0], preproc_imgs[1])
        elif method == "MSPPC_CNN":
            print("todo")
        elif method == "PPC4_CNN":
            print("todo")
        elif method == "PPC8_CNN":
            print("todo")
        elif method == "PPC12_CNN":
            print("todo")
        elif method == "PPC16_CNN":
            print("todo")
        elif method == "FVC_LBPH":
            print("todo")
        elif method == "DFC_LBPH":
            image_data["classification_data"] = get_element_to_be_classified(full_image_method, lbph_method,
                                                                             preproc_imgs[0], preproc_imgs[1])
        elif method == "MSPPC_LBPH":
            print("todo")
        elif method == "PPC4_LBPH":
            print("todo")
        elif method == "PPC8_LBPH":
            print("todo")
        elif method == "PPC12_LBPH":
            print("todo")
        elif method == "PPC16_LBPH":
            print("todo")

        data.append(image_data)

    return data


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    FVC_CNN = "FVC_CNN"
    DFC_CNN = "DFC_CNN"
    MSPPC_CNN = "MSPPC_CNN"
    PPC4_CNN = "PPC4_CNN"
    PPC8_CNN = "PPC8_CNN"
    PPC12_CNN = "PPC12_CNN"
    PPC16_CNN = "PPC16_CNN"
    FVC_LBPH = "FVC_LBPH"
    DFC_LBPH = "DFC_LBPH"
    MSPPC_LBPH = "MSPPC_LBPH"
    PPC4_LBPH = "PPC4_LBPH"
    PPC8_LBPH = "PPC8_LBPH"
    PPC12_LBPH = "PPC12_LBPH"
    PPC16_LBPH = "PPC16_LBPH"

    SHOULD_TRAIN = True
    SHOULD_TEST = False
    CURRENT_METHOD = DFC_LBPH
    CHECK_ALL_IMGS = 0
    imgs_to_be_checked = 1
    pattern_extract_genuine = re.compile('([0-9]{5})(?!\_)')
    pattern_extract_genuine_ids_from_morphed = re.compile(r'([0-9]{5})')

    all_morphed = glob.glob("../biometrix/morphed/*.jpg")
    all_genuine = glob.glob("../biometrix/genuine/*.png")

    print("Using method: {}".format(CURRENT_METHOD))

    if SHOULD_TRAIN:
        print("Training...")
        # get feature vectors for all images pairs
        data = dispatch(CURRENT_METHOD)

        # save vectors to file
        with open('../models/data/' + CURRENT_METHOD + '_data.json', 'w') as outfile:
            print("Saving {} data to file...".format(CURRENT_METHOD))
            json.dump(data, outfile, cls=NumpyEncoder)

    if SHOULD_TEST:
        print("Testing...")
        with open('../models/data/' + CURRENT_METHOD + '_data.json', 'r') as infile:
            print("Loading {} data from file...".format(CURRENT_METHOD))
            data = json.load(infile)
            print("Converting data to correct format...")
            for el in data:
                el["classification_data"] = np.asarray(el["classification_data"])
