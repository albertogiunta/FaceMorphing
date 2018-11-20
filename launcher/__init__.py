import numpy as np
from imageio import imsave

from featureVectorMerging.differential_comparison import DifferentialComparison
from launcher.apply_methods import apply_cnn, apply_lbph_to_whole_img, apply_preproc, apply_lbph_to_patched_img
from launcher.data_io import load_data_for_current_method, load_imgs_dbs, load_imgs_labels_and_csv
from utils import img_utils
from utils.base_utils import print_status

SVM_LINEAR = "LINEAR_SVM"
SVM_RBF = "RBF_SVM"
SVM_POLY = "POLY_SVM"
SVM_SIGM = "SIGM_SVM"

DIM_96 = "96"
DIM_256 = "256"
DIM_ORIGINAL = "original"
DIM_ORIGINAL_UNCROPPED = "original_uncropped"

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

DB_MORPHEDDB_DIGITAL = "morpheddbdigital"
DB_MORPHEDDB_PRINTEDSCANNED = "morpheddbprintedscanned"
DB_BIOMETIX_FERET = "biometixferet"


def save_preprocessed_originals_to_file():
    morphed, genuine4morphed, genuine = load_imgs_dbs(DIM_ORIGINAL, ALIGN_NONE, CURRENT_DB)

    for i, img in enumerate(morphed):
        print(img)
        print_status(i)
        new_img = apply_preproc(img_utils.load_img_dlib_rgb(img))[0]
        imsave(img, new_img)

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
    save_to_file(data_to_be_written, CURRENT_DB, CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)

    print("\tFinished")
    print("Created file: ../assets/json/{}.json".format(CURRENT_METHOD))


def get_fvs_no_reference():
    fvs = []
    if CURRENT_METHOD == FVC_CNN_OF:
        from launcher.morpheddb_runner import get_morphed_labels
        from launcher.morpheddb_runner import get_genuine_labels
        fvs_morphed = [(morphed_fv[i], get_morphed_labels()[i]) for i in range(len(morphed_fv))]
        fvs_genuine = [(genuine_fv[i], get_genuine_labels()[i]) for i in range(len(genuine_fv))]
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
    for i, el in enumerate(singles):
        img = el[0]
        name = el[1]
        print_status(i)
        if CURRENT_METHOD == FVC_CNN:
            fv.append((apply_cnn(img)[0], name))
        elif CURRENT_METHOD == FVC_LBPH:
            fv.append((apply_lbph_to_whole_img(img)[0], name))
    return fv


def process_pairs(pairs):
    fv = []
    for i, el in enumerate(pairs):
        morph = el[0]
        gen = el[1]
        name = el[2]
        print_status(i)
        if CURRENT_METHOD == DFC_CNN:
            feature_vectors = apply_cnn(morph, gen)
            if feature_vectors[0] is None or feature_vectors[1] is None:
                continue
            fv.append((DifferentialComparison.get_differential_fv_from_vector(feature_vectors), name))
        elif CURRENT_METHOD == DFC_LBPH:
            feature_vectors = apply_lbph_to_whole_img(morph, gen)
            if feature_vectors[0] is None or feature_vectors[1] is None:
                continue
            fv.append((DifferentialComparison.get_differential_fv_from_vector(feature_vectors), name))
        elif CURRENT_METHOD == PPC4_LBPH:
            fv.append((apply_lbph_to_patched_img((morph, gen), 4), name))
        elif CURRENT_METHOD == PPC8_LBPH:
            fv.append((apply_lbph_to_patched_img((morph, gen), 8), name))
        elif CURRENT_METHOD == PPC12_LBPH:
            fv.append((apply_lbph_to_patched_img((morph, gen), 12), name))
        elif CURRENT_METHOD == PPC16_LBPH:
            fv.append((apply_lbph_to_patched_img((morph, gen), 16), name))
    return fv


def find_and_save_best_clf_for_current_method():
    clf, feature_vectors, classes, ids = load_data_for_current_method(CURRENT_DB, CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN, TOT_TO_BE_LOADED)
    print("Grid search for method {} {} {}".format(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN))
    clf.grid_search_cross_val(feature_vectors, classes)


def calculate_frr():
    clf, feature_vectors, classes, ids = load_data_for_current_method(CURRENT_DB, CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN, TOT_TO_BE_LOADED)
    print("\nFAR/FRR for method {} {} {}".format(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN))
    clf.get_frr(feature_vectors, classes, ids)


def calculate_fusion_frr(fvc=FVC_LBPH, dfc=FVC_CNN, dim=DIM_96, align=ALIGN_EYES_LIP):
    print()
    print(fvc, dfc, dim, align)
    from sklearn.model_selection import train_test_split

    clf_a, feature_vectors_a, classes_a = load_data_for_current_method(CURRENT_DB, fvc, dim, align, TOT_TO_BE_LOADED)
    clf_b, feature_vectors_b, classes_b = load_data_for_current_method(CURRENT_DB, dfc, dim, align, TOT_TO_BE_LOADED)
    clf_a = clf_a._load_classifier()
    clf_b = clf_b._load_classifier()
    min_fv_len = min(len(feature_vectors_a), len(feature_vectors_b), TOT_TO_BE_LOADED)
    feature_vectors_a = feature_vectors_a[:min_fv_len]
    feature_vectors_b = feature_vectors_b[:min_fv_len]
    classes_a = classes_a[:min_fv_len]
    classes_b = classes_b[:min_fv_len]

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(feature_vectors_a, classes_a, test_size=0.3, random_state=42)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(feature_vectors_b, classes_b, test_size=0.3, random_state=42)

    y_probs_a = clf_a.predict_proba(X_test_a)
    y_probs_b = clf_b.predict_proba(X_test_b)

    # print("How many genuines in A: " + str(len([x for x in classes_a if x == 1])))
    # print("How many genuines in B: " + str(len([x for x in classes_b if x == 1])))
    # print("FV A: {} - CLS A: {}".format(len(feature_vectors_a), len(classes_a)))
    # print("FV B: {} - CLS B: {}".format(len(feature_vectors_b), len(classes_b)))
    # print("TRAIN A: {}, TEST A: {}".format(len(X_train_a), len(X_test_a)))
    # print("TRAIN B: {}, TEST B: {}".format(len(X_train_b), len(X_test_b)))

    # y_probs = np.mean([y_probs_a, y_probs_b], axis=0)
    y_probs = np.average([y_probs_a, y_probs_b], axis=0, weights=[2, 1])

    from SVM.metrics_utils import compute_frr_at_given_far_from_probabilities
    fars = [0.05, 0.1]
    frrs = []
    farsv = []
    for far in fars:
        frr, farv = compute_frr_at_given_far_from_probabilities(y_probs, far, y_test_a, 1)
        frrs.append(str(round(frr * 100, 2)) + "%")
        farsv.append(str(round(farv * 100, 2)) + "%")
    header = ["5.00%", "10.00%"]
    rows = [frrs, farsv]
    from tabulate import tabulate
    print(tabulate(rows, headers=header, floatfmt=".2f"))


def run_all_fusion():
    # extractions = [[FVC_LBPH, DFC_LBPH], [FVC_CNN, DFC_CNN], [FVC_CNN_OF, DFC_CNN_OF]]
    extractions = [[FVC_LBPH, FVC_CNN], [DFC_LBPH, DFC_CNN]]

    dimensions = [DIM_ORIGINAL, DIM_96, DIM_256]
    alignments = [ALIGN_EYES_NOSE, ALIGN_EYES_LIP]
    for extr in extractions:
        if extr == [FVC_CNN_OF, DFC_CNN_OF]:
            for align in alignments:
                calculate_fusion_frr(extr[0], extr[1], DIM_96, align)
        else:
            for dim in dimensions:
                if dim == DIM_ORIGINAL:
                    calculate_fusion_frr(extr[0], extr[1], dim, ALIGN_NONE)
                else:
                    for align in alignments:
                        calculate_fusion_frr(extr[0], extr[1], dim, align)


def run_all():
    # methods = [FVC_LBPH, FVC_CNN, DFC_LBPH, DFC_CNN, PPC4_LBPH, PPC8_LBPH, PPC12_LBPH, PPC16_LBPH]
    extractions = [FVC_LBPH, DFC_LBPH, FVC_CNN, DFC_CNN, FVC_CNN_OF, DFC_CNN_OF]
    # extractions = [DFC_LBPH]
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
    # find_and_save_best_clf_for_current_method()
    calculate_frr()

result = []
CURRENT_DB = DB_MORPHEDDB_DIGITAL

if CURRENT_DB == DB_MORPHEDDB_DIGITAL or CURRENT_DB == DB_MORPHEDDB_PRINTEDSCANNED:
    MAX_MORPHED = 100
    MAX_GENUINES = 327
elif CURRENT_DB == DB_BIOMETIX_FERET:
    MAX_MORPHED = 900
    MAX_GENUINES = 900
else:
    MAX_MORPHED = 0
    MAX_GENUINES = 0

TOT_TO_BE_LOADED = MAX_MORPHED + MAX_GENUINES

CURRENT_METHOD = FVC_LBPH
CURRENT_DIM = DIM_ORIGINAL
CURRENT_ALIGN = ALIGN_NONE

if CURRENT_DIM == DIM_ORIGINAL:
    CURRENT_ALIGN = ALIGN_NONE
if CURRENT_METHOD == DFC_CNN_OF or CURRENT_METHOD == FVC_CNN_OF:
    CURRENT_DIM = DIM_96

morphed, genuine4morphed, genuine = load_imgs_dbs(CURRENT_DIM, CURRENT_ALIGN, CURRENT_DB)

morphed_labels, morphed_fv, sorted_morphed_labels_fvs, \
genuine4morphed_labels, genuine4morphed_fv, sorted_genuine4morphed_labels_fvs, \
genuine_labels, genuine_fv, sorted_genuine_labels_fvs = load_imgs_labels_and_csv(CURRENT_DIM, CURRENT_ALIGN, CURRENT_DB)

if __name__ == '__main__':
    # save_preprocessed_originals_to_file()

    feature_extraction_to_json()
    find_and_save_best_clf_for_current_method()
    calculate_frr()

    # calculate_fusion_frr()
    # run_all_fusion()

    # run_all()

    pass
