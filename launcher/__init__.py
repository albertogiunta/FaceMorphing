import glob
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

    print("Processing images:\n"
          "Genuine (Base) : {}\n"
          "Morphed: {} + {}\n"
          .format(base_genuine_id, genuine_ids[0], genuine_ids[1]))

    return genuine_path, morphed_path


def apply_preproc(genuine_path, morphed_path):
    preproc_imgs = []
    preproc = Preprocessing()

    preproc_imgs.append(preproc.preprocess_img_path(genuine_path))
    preproc_imgs.append(preproc.preprocess_img_path(morphed_path))

    return preproc_imgs


def apply_cnn(genuine, morphed, use_blocks=False):
    feature_vectors = []

    genuine = skimage.img_as_ubyte(genuine)
    morphed = skimage.img_as_ubyte(morphed)

    cnn = CNNFeatureVectorExtraction()
    feature_vectors.append(cnn.get_img_descriptor_from_img(genuine, use_blocks))
    feature_vectors.append(cnn.get_img_descriptor_from_img(morphed, use_blocks))

    return feature_vectors


def apply_lbph(genuine, morphed, use_blocks=True):
    feature_vectors = []

    genuine = rgb2gray(genuine)
    morphed = rgb2gray(morphed)

    lbph = LBPHFeatureVectorExtraction()
    feature_vectors.append(lbph.get_img_descriptor_from_img(genuine, use_blocks))
    feature_vectors.append(lbph.get_img_descriptor_from_img(morphed, use_blocks))

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
        print(np.shape(genuine))
        if fv_extraction_method == "cnn":
            feature_vectors = apply_cnn(genuine, morphed)
        elif fv_extraction_method == "lbph":
            feature_vectors = apply_lbph(genuine, morphed)
        differential = DifferentialComparison.get_differential_fv(feature_vectors[1], feature_vectors[0])
        # print(differential)
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
                # print(np.shape(genuine_patch))
                if fv_extraction_method == "cnn":
                    feature_vectors = apply_cnn(genuine_patch, morphed_patch, True)
                elif fv_extraction_method == "lbph":
                    feature_vectors = apply_lbph(genuine_patch, morphed_patch, False)

                euclidean = DifferentialComparison.calculate_euclidean_distance(feature_vectors[0], feature_vectors[1])
                concatenated.append(euclidean)

        return concatenated


def show_imgs(imgs):
    io.imshow_collection(imgs)
    io.show()


if __name__ == '__main__':

    imgs_to_be_checked = 1
    pattern_extract_genuine = re.compile('([0-9]{5})(?!\_)')
    pattern_extract_genuine_ids_from_morphed = re.compile(r'([0-9]{5})')

    all_morphed = glob.glob("../biometrix/morphed/*.jpg")
    all_genuine = glob.glob("../biometrix/genuine/*.png")

    for i, img in enumerate(all_morphed):
        if i >= imgs_to_be_checked:
            break

        genuine_path, morphed_path = get_path_of_morphed_and_base_genuine(img)

        # preprocess
        # 1 genuine
        # 2 morphed
        preproc_imgs = apply_preproc(genuine_path, morphed_path)

        full_image_method = "full_image"
        patch_wise_method = "patch_wise"
        lbph_method = "lbph"
        cnn_method = "cnn"

        # vector = get_element_to_be_classified(full_image_method, lbph_method, preproc_imgs[0], preproc_imgs[1])
        vector = get_element_to_be_classified(full_image_method, cnn_method, preproc_imgs[0], preproc_imgs[1])
        # vector = get_element_to_be_classified(patch_wise_method, lbph_method, preproc_imgs[0], preproc_imgs[1])
        # vector = get_element_to_be_classified(patch_wise_method, cnn_method, preproc_imgs[0], preproc_imgs[1])

        # show_imgs(preproc_imgs)

        # SVM
        prediction = classify(vector)
        print(prediction)
