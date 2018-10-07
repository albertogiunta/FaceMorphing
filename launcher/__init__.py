import json

import numpy as np
import skimage
from skimage.color import rgb2gray
from skimage.util import view_as_blocks

from featureVectorMerging.differential_comparison import DifferentialComparison
from utils.config_utils import NumpyEncoder

FVC_CNN = "FVC_CNN"
FVC_LBPH = "FVC_LBPH"

DFC_CNN = "DFC_CNN"
DFC_LBPH = "DFC_LBPH"

PPC4_LBPH = "PPC4_LBPH"
PPC8_LBPH = "PPC8_LBPH"
PPC12_LBPH = "PPC12_LBPH"
PPC16_LBPH = "PPC16_LBPH"

# TODO implement this
MSPPC_CNN = "MSPPC_CNN"
MSPPC_LBPH = "MSPPC_LBPH"


# data = {
#     'fv': None,
#     "cls": None
# }


def apply_preproc(*args):
    from preprocessing.preprocessing import Preprocessing
    preproc = Preprocessing()
    preproc_imgs = []

    for img in args:
        preproc_imgs.append(preproc.preprocess_img(img))

    return preproc_imgs


def apply_cnn(*args):
    from featureExtraction.cnn_fv_extraction import CNNFeatureVectorExtraction
    cnn = CNNFeatureVectorExtraction()
    feature_vectors = []

    for img in args:
        temp_img = skimage.img_as_ubyte(img)
        feature_vectors.append(cnn.get_img_descriptor_from_img(temp_img))

    return feature_vectors


def apply_lbph_to_whole_img(*args):
    from featureExtraction.lbph_fv_extraction import LBPHFeatureVectorExtraction
    from skimage.color import rgb2gray
    lbph = LBPHFeatureVectorExtraction()
    feature_vectors = []

    for img in args:
        temp_img = rgb2gray(img)
        feature_vectors.append(lbph.get_img_descriptor_from_img(temp_img))

    return feature_vectors


def apply_lbph_to_patch(*args):
    from featureExtraction.lbph_fv_extraction import LBPHFeatureVectorExtraction
    from skimage.color import rgb2gray
    lbph = LBPHFeatureVectorExtraction()
    feature_vectors = []

    for img in args:
        temp_img = rgb2gray(img)
        feature_vectors.append(lbph.get_img_descriptor_from_patch(temp_img))

    return feature_vectors


def get_patched_lbph_fv(img_pair, grid_size):
    temp_result = []

    img_pair[0] = rgb2gray(img_pair[0])
    img_pair[1] = rgb2gray(img_pair[1])

    img_size = 256
    margin = 256 % grid_size
    if margin != 0:
        if margin % 2 == 0:
            edge_crop = int(margin / 2)
            for i, img in enumerate(img_pair):
                img_pair[i] = skimage.util.crop(img, edge_crop)
        else:
            print("Image size and grid size are REALLY not compatible.")
            exit(-1)

    patch_shape = (int((img_size - margin) / grid_size), int((img_size - margin) / grid_size))
    patched_pair = [view_as_blocks(img_pair[0], patch_shape),
                    view_as_blocks(img_pair[1], patch_shape)]

    for i, row in enumerate(patched_pair[0]):
        for j, col in enumerate(patched_pair[0][i]):
            patch_pair = [patched_pair[0][i, j], patched_pair[1][i, j]]
            feature_vectors = apply_lbph_to_patch(patch_pair[0], patch_pair[1])

            euclidean = DifferentialComparison.calculate_euclidean_distance(feature_vectors[0],
                                                                            feature_vectors[1])
            temp_result.append(euclidean)

    return temp_result


def process_singles(singles):
    result = []
    for i, img in enumerate(singles):
        if i == 0 or (i + 1) % 10 == 0:
            print_status(i)
        preprocessed_img = apply_preproc(img)[0]
        if CURRENT_METHOD == FVC_CNN:
            result.append(apply_cnn(preprocessed_img)[0])
        elif CURRENT_METHOD == FVC_LBPH:
            result.append(apply_lbph_to_whole_img(preprocessed_img)[0])
    return result


def process_pairs(pairs):
    result = []
    for i, pair in enumerate(pairs):
        if i == 0 or (i + 1) % 10 == 0:
            print_status(i)
        preprocessed_pair = apply_preproc(pair[0], pair[1])
        if CURRENT_METHOD == DFC_CNN:
            feature_vectors = apply_cnn(preprocessed_pair[0], preprocessed_pair[1])
            result.append(DifferentialComparison.get_differential_fv_from_vector(feature_vectors))
        elif CURRENT_METHOD == DFC_LBPH:
            feature_vectors = apply_lbph_to_whole_img(preprocessed_pair[0], preprocessed_pair[1])
            result.append(DifferentialComparison.get_differential_fv_from_vector(feature_vectors))
        elif CURRENT_METHOD == PPC4_LBPH:
            result.append(get_patched_lbph_fv(preprocessed_pair, 4))
        elif CURRENT_METHOD == PPC8_LBPH:
            result.append(get_patched_lbph_fv(preprocessed_pair, 8))
        elif CURRENT_METHOD == PPC12_LBPH:
            result.append(get_patched_lbph_fv(preprocessed_pair, 12))
        elif CURRENT_METHOD == PPC16_LBPH:
            result.append(get_patched_lbph_fv(preprocessed_pair, 16))

    return result


def print_status(i):
    from time import gmtime, strftime
    if i != 0:
        i += 1
    print("\t\t\t\t{} - {}".format(i, strftime("%Y-%m-%d %H:%M:%S", gmtime())))


def process_morphed_singles():
    from launcher.biometix_runner import get_morphed_images
    imgs = get_morphed_images(IMGS_TO_BE_PROCESSED)
    return process_singles(imgs)


def process_genuine_singles():
    from launcher.biometix_runner import get_genuine_images
    imgs = get_genuine_images(IMGS_TO_BE_PROCESSED)
    return process_singles(imgs)


def process_morphed_pairs():
    from launcher.biometix_runner import get_morphed_genuine_pairs
    pairs = get_morphed_genuine_pairs(IMGS_TO_BE_PROCESSED)
    return process_pairs(pairs)


def process_genuine_pairs():
    from launcher.feret_runner import get_genuine_genuine_pairs
    pairs = get_genuine_genuine_pairs(IMGS_TO_BE_PROCESSED)
    return process_pairs(pairs)


def extract_data_with_current_method():
    print("Processing images with method: {}".format(CURRENT_METHOD))

    print("\tTraining")

    data_to_be_written = []

    if CURRENT_METHOD == FVC_CNN or CURRENT_METHOD == FVC_LBPH:
        print("\t\tProcessing genuine data")
        data_to_be_written.append(get_data_to_be_written(process_genuine_singles(), 1))
        print("\t\tProcessing attack data")
        data_to_be_written.append(get_data_to_be_written(process_morphed_singles(), -1))
    else:
        print("\t\tProcessing genuine data")
        data_to_be_written.append(get_data_to_be_written(process_genuine_pairs(), 1))
        print("\t\tProcessing attack data")
        data_to_be_written.append(get_data_to_be_written(process_morphed_pairs(), -1))

    data_to_be_written = np.array(data_to_be_written).flatten()

    print("\tWriting data to file (data length: {})".format(len(data_to_be_written)))
    save_to_file(data_to_be_written)

    print("\tFinished")
    print("Created file: ../assets/data/{}_data.json".format(CURRENT_METHOD))


def get_data_to_be_written(feature_vectors, cls):
    result = []
    for fv in feature_vectors:
        data = {'fv': fv, 'cls': cls}
        result.append(data)
    return result


def save_to_file(data):
    with open('../assets/data/' + CURRENT_METHOD + '_data.json', 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)


def train_svm_with_current_method():
    print("Training with {}".format(CURRENT_METHOD))
    with open('../assets/data/' + CURRENT_METHOD + '_data.json', 'r') as infile:
        data = json.load(infile)
        print("\tLoaded data from file...")

        fvs = []
        clss = []

        for el in data:
            fvs.append(el['fv'])
            clss.append(el['cls'])

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(fvs, clss, test_size=0.70)

        from SVM.svm_classification import SVMClassifier
        svm = SVMClassifier(CURRENT_METHOD)
        svm.train(X_train, y_train)

        print(svm.get_score(X_test, y_test))


# def predict_result():
#     from SVM.svm_classification import SVMClassifier
#     svm = SVMClassifier(CURRENT_METHOD)
#
#     from launcher.biometix_runner import get_nth_morphed_image, get_nth_genuine_image
#     from launcher.feret_runner import get_nth_genuine_image2
#     interval = 15
#     test_morphed_img = get_nth_morphed_image(IMGS_TO_BE_PROCESSED + interval)
#     test_genuine_img = get_nth_genuine_image(IMGS_TO_BE_PROCESSED + interval)
#     test_genuine2_img = get_nth_genuine_image2(IMGS_TO_BE_PROCESSED + interval)
#
#     # img = test_morphed_img
#     img = test_genuine_img
#     # img = test_genuine2_img
#
#     test_preproc_img = apply_preproc(img)[0]
#     test_fv = None
#     if CURRENT_METHOD == FVC_CNN:
#         test_fv = apply_cnn(test_preproc_img)[0]
#     elif CURRENT_METHOD == FVC_LBPH:
#         test_fv = apply_lbph_to_whole_img(test_preproc_img)[0]
#
#     result = svm.predict_class(test_fv)
#     print(result)


if __name__ == '__main__':
    process_all_imgs = 0
    IMGS_TO_BE_PROCESSED = 900

    CURRENT_METHOD = FVC_LBPH

    extract_data_with_current_method()
    # train_svm_with_current_method()
    # predict_result()
