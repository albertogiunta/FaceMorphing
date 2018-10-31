import json

import numpy as np
import skimage
from skimage.color import rgb2gray
from skimage.util import view_as_blocks

from featureVectorMerging.differential_comparison import DifferentialComparison
from utils.config_utils import NumpyEncoder

global CURRENT_METHOD

SVM_LINEAR = "LINEAR_SVM"
SVM_RBF = "RBF_SVM"
SVM_POLY = "POLY_SVM"
SVM_SIGM = "SIGM_SVM"

FVC_CNN = "FVC_CNN"
FVC_CNN_OF = "FVC_CNN_OF"
FVC_LBPH = "FVC_LBPH"

DFC_CNN = "DFC_CNN"
DFC_CNN_OF = "DFC_CNN_OF"
DFC_LBPH = "DFC_LBPH"

PPC4_LBPH = "PPC4_LBPH"
PPC8_LBPH = "PPC8_LBPH"
PPC12_LBPH = "PPC12_LBPH"
PPC16_LBPH = "PPC16_LBPH"

# TODO implement this
MSPPC_CNN = "MSPPC_CNN"
MSPPC_LBPH = "MSPPC_LBPH"

FVC_CNN_TEST = "FVC_CNN_TEST"
FVC_LBPH_TEST = "FVC_LBPH_TEST"


def apply_preproc(*args):
    from preprocessing.preprocessing import Preprocessing
    preproc = Preprocessing()
    preproc_imgs = []

    for img in args:
        new_img = preproc.preprocess_img(img)
        if new_img is not None:
            preproc_imgs.append(new_img)
    return preproc_imgs


def apply_cnn(*args):
    from featureExtraction.cnn_fv_extraction import CNNFeatureVectorExtraction
    cnn = CNNFeatureVectorExtraction()
    feature_vectors = []

    for img in args:
        feature_vectors.append(cnn.get_img_descriptor_from_img(img))

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

            euclidean = DifferentialComparison.calculate_euclidean_distance_from_binary_vector(feature_vectors)
            temp_result.append(euclidean)

    return temp_result


def process_singles(singles):
    result = []
    for i, img in enumerate(singles):
        print_status(i)
        # new_img = apply_preproc(img)
        new_img = [img]
        if len(new_img) > 0:
            preprocessed_img = new_img[0]
            if CURRENT_METHOD == FVC_CNN or CURRENT_METHOD == FVC_CNN_TEST:
                result.append(apply_cnn(preprocessed_img)[0])
            elif CURRENT_METHOD == FVC_LBPH or CURRENT_METHOD == FVC_LBPH_TEST:
                result.append(apply_lbph_to_whole_img(preprocessed_img)[0])
    return result


def process_pairs(pairs):
    result = []
    for i, pair in enumerate(pairs):
        print_status(i)
        preprocessed_pair = apply_preproc(pair[0], pair[1])
        if len(preprocessed_pair) != 2:
            continue
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
    if i == 0 or (i + 1) % 10 == 0:
        from time import gmtime, strftime
        if i != 0:
            i += 1
        print("\t\t\t\t{} - {}".format(i, strftime("%Y-%m-%d %H:%M:%S", gmtime())))


def process_morphed_singles():
    if CURRENT_METHOD == FVC_CNN_OF:
        from launcher.openface_runner import biometix_morphed_fv
        fvs = biometix_morphed_fv[0:IMGS_TO_BE_PROCESSED]
        return fvs
    else:
        from launcher.biometix_runner import get_morphed_images
        imgs = get_morphed_images(IMGS_TO_BE_PROCESSED)
        return process_singles(imgs)


def process_genuine_singles():
    if CURRENT_METHOD == FVC_CNN_OF:
        from launcher.openface_runner import feret_fv
        fvs = feret_fv[0:IMGS_TO_BE_PROCESSED]
        return fvs
    else:
        from launcher.feret_runner import get_genuine_images
        imgs = get_genuine_images(IMGS_TO_BE_PROCESSED)
        return process_singles(imgs)


def process_morphed_pairs():
    if CURRENT_METHOD == DFC_CNN_OF:
        from launcher.openface_runner import get_morphed_genuine_diff_fv
        morphed_pairs = get_morphed_genuine_diff_fv(IMGS_TO_BE_PROCESSED)
        return morphed_pairs
    else:
        from launcher.biometix_runner import get_morphed_genuine_pairs
        pairs = get_morphed_genuine_pairs(IMGS_TO_BE_PROCESSED)
        return process_pairs(pairs)


def process_genuine_pairs():
    if CURRENT_METHOD == DFC_CNN_OF:
        from launcher.openface_runner import get_genuine_genuine_diff_fv
        genuine_pairs = get_genuine_genuine_diff_fv(IMGS_TO_BE_PROCESSED)
        return genuine_pairs
    else:
        from launcher.feret_runner import get_genuine_genuine_pairs
        pairs = get_genuine_genuine_pairs(IMGS_TO_BE_PROCESSED)
        return process_pairs(pairs)


def extract_data_with_current_method():
    print("Processing images with method: {}".format(CURRENT_METHOD))

    print("\tTraining")

    data_to_be_written = []

    if CURRENT_METHOD == FVC_CNN or CURRENT_METHOD == FVC_LBPH or CURRENT_METHOD == FVC_CNN_OF or CURRENT_METHOD == FVC_CNN_TEST or CURRENT_METHOD == FVC_LBPH_TEST:
        print("\t\tProcessing genuine data")
        data_to_be_written.append(get_data_to_be_written(process_genuine_singles(), 1))
        print("\t\tProcessing attack data")
        data_to_be_written.append(get_data_to_be_written(process_morphed_singles(), 0))
    else:
        print("\t\tProcessing genuine data")
        data_to_be_written.append(get_data_to_be_written(process_genuine_pairs(), 1))
        print("\t\tProcessing attack data")
        data_to_be_written.append(get_data_to_be_written(process_morphed_pairs(), 0))

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


def get_svm_classifier():
    from SVM.svm_classification import SVMClassifier
    svm_classifier = SVMClassifier(CURRENT_METHOD, CURRENT_SVM)
    return svm_classifier


def load_data_for_current_method():
    print("Loading data from file")
    with open('../assets/data/' + CURRENT_METHOD + '_data.json', 'r') as infile:
        data = json.load(infile)
        print("\tLoaded")

        feature_vectors = []
        classes = []

        if len(data) == 2:
            for cls in data:
                for el in cls:
                    feature_vectors.append(el['fv'])
                    classes.append(el['cls'])
        else:
            for el in data:
                feature_vectors.append(el['fv'])
                classes.append(el['cls'])

        print("\tData structures created - {} samples loaded".format(len(feature_vectors)))
        return feature_vectors, classes


def find_and_save_best_clf_for_current_method():
    feature_vectors, classes = load_data_for_current_method()
    print("Grid search for method {}".format(CURRENT_METHOD))
    get_svm_classifier().grid_search_cross_val(feature_vectors, classes)


def calculate_bpcer_for_current_method():
    feature_vectors, classes = load_data_for_current_method()
    print("APCER/BPCER for method {}".format(CURRENT_METHOD))
    get_svm_classifier().get_bpcer(feature_vectors, classes)


def calculate_far():
    feature_vectors, classes = load_data_for_current_method()
    print("FAR/FRR for method {}".format(CURRENT_METHOD))
    get_svm_classifier().get_frr(feature_vectors, classes)


def find_false_negatives():
    feature_vectors, classes = load_data_for_current_method()
    print("APCER/BPCER for method {}".format(CURRENT_METHOD))
    get_svm_classifier().find_false_negatives(feature_vectors, classes)


def execute_f_for_all_methods():
    methods = [FVC_LBPH, FVC_CNN, DFC_LBPH, DFC_CNN, PPC4_LBPH, PPC8_LBPH, PPC12_LBPH, PPC16_LBPH]
    for method in methods:
        global CURRENT_METHOD
        CURRENT_METHOD = method
        calculate_far()
        print()
        print()


if __name__ == '__main__':
    result = []

    process_all_imgs = 0
    IMGS_TO_BE_PROCESSED = 900

    global CURRENT_METHOD
    CURRENT_METHOD = DFC_LBPH
    CURRENT_SVM = SVM_LINEAR

    extract_data_with_current_method()
    find_and_save_best_clf_for_current_method()
    calculate_far()

    # find_false_negatives()
    # execute_f_for_all_methods()
