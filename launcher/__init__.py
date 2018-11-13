import numpy as np

from featureVectorMerging.differential_comparison import DifferentialComparison
from launcher.apply_methods import apply_cnn, apply_lbph_to_whole_img, apply_preproc, apply_lbph_to_patched_img
from launcher.data_io import load_data_for_current_method, load_imgs_dbs, load_imgs_labels_and_csv
from utils.base_utils import print_status

SVM_LINEAR = "LINEAR_SVM"
SVM_RBF = "RBF_SVM"
SVM_POLY = "POLY_SVM"
SVM_SIGM = "SIGM_SVM"

DIM_96 = "96"
DIM_256 = "256"
DIM_ORIGINAL = "original"

ALIGN_EYES_NOSE = "eyesnose"
ALIGN_EYES_LIP = "eyeslip"
ALIGN_NONE = ""

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


def feature_extraction_to_json():
    from launcher.data_io import get_data_to_be_written
    from launcher.data_io import save_to_file
    print("Processing images with method: {} {} {}".format(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN))
    print("\tTraining")
    data_to_be_written = []

    if CURRENT_METHOD == FVC_CNN or CURRENT_METHOD == FVC_LBPH or CURRENT_METHOD == FVC_CNN_OF:
        fvs_morphed_genuine = get_fvs_no_reference()
        data_to_be_written.append(get_data_to_be_written(fvs_morphed_genuine[0], 0))
        data_to_be_written.append(get_data_to_be_written(fvs_morphed_genuine[1], 1))
    else:
        fvs_morphed_genuine = get_fvs_differential()
        data_to_be_written.append(get_data_to_be_written(fvs_morphed_genuine[0], 0))
        data_to_be_written.append(get_data_to_be_written(fvs_morphed_genuine[1], 1))

    for i, el in enumerate(data_to_be_written):
        if el[0] is None or len(el[0]) < 1:
            data_to_be_written.pop(i)

    print("\tWriting data to file (data length: {} {})".format(len(data_to_be_written[0]), len(data_to_be_written[1])))
    data_to_be_written = np.array(data_to_be_written).flatten()
    save_to_file(data_to_be_written, CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)

    print("\tFinished")
    print("Created file: ../assets/json/{}.json".format(CURRENT_METHOD))


def get_fvs_no_reference():
    fvs = []
    if CURRENT_METHOD == FVC_CNN_OF:
        fvs_morphed = morphed_fv
        fvs_genuine = genuine_fv
    else:
        # from launcher.biometix_runner import get_morphed_images
        # from launcher.feret_runner import get_genuine_images
        from launcher.morpheddb_runner import get_morphed_images
        from launcher.morpheddb_runner import get_genuine_images
        fvs_morphed = process_singles(get_morphed_images())
        fvs_genuine = process_singles(get_genuine_images())

    fvs.append(fvs_morphed)
    fvs.append(fvs_genuine)
    return fvs


def get_fvs_differential():
    fvs = []
    if CURRENT_METHOD == DFC_CNN_OF:
        # from launcher.openface_runner import get_morphed_genuine_diff_fv
        # from launcher.openface_runner import get_genuine_genuine_diff_fv
        from launcher.morpheddb_runner import get_morphed_genuine_diff_fv
        from launcher.morpheddb_runner import get_genuine_genuine_diff_fv
        fvs_morphed = get_morphed_genuine_diff_fv()
        fvs_genuine = get_genuine_genuine_diff_fv()
    else:
        # from launcher.biometix_runner import get_morphed_genuine_pairs
        # from launcher.feret_runner import get_genuine_genuine_pairs
        from launcher.morpheddb_runner import get_morphed_genuine_pairs
        from launcher.morpheddb_runner import get_genuine_genuine_pairs
        fvs_morphed = process_pairs(get_morphed_genuine_pairs())
        fvs_genuine = process_pairs(get_genuine_genuine_pairs())

    fvs.append(fvs_morphed)
    fvs.append(fvs_genuine)
    return fvs


def process_singles(singles):
    fv = []
    for i, img in enumerate(singles):
        print_status(i)
        new_img = [img]
        if len(new_img) > 0:
            preprocessed_img = new_img[0]
            if CURRENT_METHOD == FVC_CNN:
                fv.append(apply_cnn(preprocessed_img)[0])
            elif CURRENT_METHOD == FVC_LBPH:
                fv.append(apply_lbph_to_whole_img(preprocessed_img)[0])
    return fv


def process_pairs(pairs):
    fv = []
    for i, pair in enumerate(pairs):
        print_status(i)
        preprocessed_pair = [pair[0], pair[1]]
        if len(preprocessed_pair) != 2:
            continue
        if CURRENT_METHOD == DFC_CNN:
            feature_vectors = apply_cnn(preprocessed_pair[0], preprocessed_pair[1])
            if feature_vectors[0] is None or feature_vectors[1] is None:
                continue
            fv.append(DifferentialComparison.get_differential_fv_from_vector(feature_vectors))
        elif CURRENT_METHOD == DFC_LBPH:
            feature_vectors = apply_lbph_to_whole_img(preprocessed_pair[0], preprocessed_pair[1])
            if feature_vectors[0] is None or feature_vectors[1] is None:
                continue
            fv.append(DifferentialComparison.get_differential_fv_from_vector(feature_vectors))
        elif CURRENT_METHOD == PPC4_LBPH:
            fv.append(apply_lbph_to_patched_img(preprocessed_pair, 4))
        elif CURRENT_METHOD == PPC8_LBPH:
            fv.append(apply_lbph_to_patched_img(preprocessed_pair, 8))
        elif CURRENT_METHOD == PPC12_LBPH:
            fv.append(apply_lbph_to_patched_img(preprocessed_pair, 12))
        elif CURRENT_METHOD == PPC16_LBPH:
            fv.append(apply_lbph_to_patched_img(preprocessed_pair, 16))
    return fv


def find_and_save_best_clf_for_current_method():
    clf, feature_vectors, classes = load_data_for_current_method(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)
    print("Grid search for method {} {} {}".format(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN))
    clf.grid_search_cross_val(feature_vectors, classes)


def calculate_far():
    clf, feature_vectors, classes = load_data_for_current_method(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)
    print("FAR/FRR for method {} {} {}".format(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN))
    clf.get_frr(feature_vectors, classes)
    # clf.get_bpcer(feature_vectors, classes)


def run_all():
    # methods = [FVC_LBPH, FVC_CNN, DFC_LBPH, DFC_CNN, PPC4_LBPH, PPC8_LBPH, PPC12_LBPH, PPC16_LBPH]
    # extractions = [FVC_LBPH, DFC_LBPH, FVC_CNN, DFC_CNN, FVC_CNN_OF, DFC_CNN_OF]
    extractions = [DFC_LBPH]
    dimensions = [DIM_ORIGINAL, DIM_96, DIM_256]
    alignments = [ALIGN_EYES_NOSE, ALIGN_EYES_LIP]

    global CURRENT_METHOD
    global CURRENT_ALIGN
    global CURRENT_DIM

    for method in extractions:
        CURRENT_METHOD = method
        if CURRENT_METHOD == FVC_CNN_OF or CURRENT_METHOD == DFC_CNN_OF:
            CURRENT_DIM = DIM_96
            for alignment in alignments:
                CURRENT_ALIGN = alignment
                do_stuff()
        else:
            for dimension in dimensions:
                CURRENT_DIM = dimension
                if CURRENT_DIM == DIM_ORIGINAL:
                    CURRENT_ALIGN = ALIGN_NONE
                    do_stuff()
                else:
                    for alignment in alignments:
                        CURRENT_ALIGN = alignment
                        do_stuff()


def do_stuff():
    print(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)
    # feature_extraction_to_json()
    find_and_save_best_clf_for_current_method()
    calculate_far()


result = []
IMGS_TO_BE_PROCESSED = 100

CURRENT_METHOD = DFC_CNN

CURRENT_DIM = DIM_ORIGINAL
CURRENT_ALIGN = ALIGN_EYES_LIP

if CURRENT_DIM == DIM_ORIGINAL:
    CURRENT_ALIGN = ALIGN_NONE
if CURRENT_METHOD == DFC_CNN_OF or CURRENT_METHOD == FVC_CNN_OF:
    CURRENT_DIM = DIM_96

morphed, genuine4morphed, genuine = load_imgs_dbs(CURRENT_DIM, CURRENT_ALIGN)

morphed_labels, morphed_fv, sorted_morphed_labels_fvs, \
genuine4morphed_labels, genuine4morphed_fv, sorted_genuine4morphed_labels_fvs, \
genuine_labels, genuine_fv, sorted_genuine_labels_fvs = load_imgs_labels_and_csv(CURRENT_DIM, CURRENT_ALIGN)

if __name__ == '__main__':
    # feature_extraction_to_json()
    # find_and_save_best_clf_for_current_method()
    # calculate_far()
    run_all()

    # for i, img in enumerate(morphed):
    #     print(img)
    #     print_status(i)
    #     new_img = apply_preproc(img_utils.load_img_dlib_rgb(img))[0]
    #     imsave(img, new_img)

    # for i, img in enumerate(genuine):
    #     print(img)
    #     print_status(i)
    #     new_img = apply_preproc(img_utils.load_img_dlib_rgb(img))
    #     if len(new_img) >= 1:
    #         imsave(img, new_img[0])
    #
    # for i, img in enumerate(genuine4morphed):
    #     print(img)
    #     print_status(i)
    #     new_img = apply_preproc(img_utils.load_img_dlib_rgb(img))
    #     if len(new_img) >= 1:
    #         imsave(img, new_img[0])
