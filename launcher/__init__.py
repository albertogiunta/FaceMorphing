import numpy as np
from imageio import imsave

from featureVectorMerging.differential_comparison import DifferentialComparison
from launcher.apply_methods import apply_cnn, apply_lbph_to_whole_img, apply_preproc, apply_lbph_to_patched_img
from launcher.data_io import load_data_for_current_method, load_imgs_dbs, load_imgs_labels_and_csv
from utils import img_utils
from utils.base_utils import print_status

TASK_TRAINING = "training"
TASK_TESTING = "testing"

IMG_TYPE_DIGITAL = "digital"
IMG_TYPE_PS = "ps"
IMG_TYPE_BOTH = "both"

DB_MORPHEDDB = "morphedDB"
DB_PMDB = "pmDB"
DB_BIOMETIX = "biometix"

DIM_96 = "96"
DIM_256 = "256"

ALIGN_DLIB = "dlib"
ALIGN_EYES_NOSE = "eyesnose"
ALIGN_EYES_LIP = "eyeslip"

FVC_LBPH = "FVC_LBPH"
FVC_CNN = "FVC_CNN"
FVC_CNN_OF = "FVC_CNN_OF"

DFC_LBPH = "DFC_LBPH"
DFC_CNN = "DFC_CNN"
DFC_CNN_OF = "DFC_CNN_OF"

PPC4_LBPH = "PPC4_LBPH"
PPC8_LBPH = "PPC8_LBPH"
PPC12_LBPH = "PPC12_LBPH"
PPC16_LBPH = "PPC16_LBPH"


def save_preprocessed_originals_to_file():
    morphed, genuine4morphed, genuine = load_imgs_dbs(build_string_db())
    # morphed, genuine4morphed, genuine = load_imgs_dbs("testing/digital/biometix/raw")

    do_mor = False
    do_gen = True
    do_g4m = False

    if do_mor:
        for i, img in enumerate(morphed):
            print(img)
            print_status(i)
            new_img = apply_preproc(img_utils.load_img_dlib_rgb(img))[0]
            # from skimage.transform import resize
            # new_img = resize(img_utils.load_img_dlib_rgb(img), (256, 256))
            imsave(img, new_img)

    if do_gen:
        for i, img in enumerate(genuine):
            print(img)
            print_status(i)
            new_img = apply_preproc(img_utils.load_img_dlib_rgb(img))
            if len(new_img) >= 1:
                imsave(img, new_img[0])

    if do_g4m:
        for i, img in enumerate(genuine4morphed):
            print(img)
            print_status(i)
            new_img = apply_preproc(img_utils.load_img_dlib_rgb(img))
            if len(new_img) >= 1:
                imsave(img, new_img[0])

def feature_extraction_to_json():
    from launcher.data_io import get_data_to_be_written
    from launcher.data_io import save_to_file
    print("Processing images with method: {} {}".format(build_string_db(), CURRENT_METHOD))
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

    print("\tWriting data to file (data length: {} morphed + {} genuines)".format(len(data_to_be_written[0]), len(data_to_be_written[1])))
    data_to_be_written = np.array(data_to_be_written).flatten()
    save_to_file(data_to_be_written, build_string_db_for_json(), CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)

    print("Created JSON feature vectors file for {} {}\n\n".format(build_string_db(), CURRENT_METHOD))


def get_fvs_no_reference():
    fvs = []
    if CURRENT_METHOD == FVC_CNN_OF:
        # TODO
        from launcher.morpheddb_runner import get_morphed_labels
        from launcher.morpheddb_runner import get_genuine_labels
        fvs_morphed = [(morphed_fv[i], get_morphed_labels(morphed)[i]) for i in range(len(morphed_fv))]
        fvs_genuine = [(genuine_fv[i], get_genuine_labels(genuine)[i]) for i in range(len(genuine_fv))]
    else:
        from launcher.morpheddb_runner import get_morphed_images
        from launcher.morpheddb_runner import get_genuine_images
        fvs_morphed = process_singles(get_morphed_images(morphed))
        fvs_genuine = process_singles(get_genuine_images(genuine))

    fvs.append(fvs_morphed)
    fvs.append(fvs_genuine)
    return fvs


def get_fvs_differential():
    fvs = []
    from launcher.morpheddb_runner import get_morphed_genuine_pairs
    from launcher.morpheddb_runner import get_genuine_genuine_pairs
    if CURRENT_METHOD == DFC_CNN_OF:
        fvs_morphed = get_morphed_genuine_pairs(sorted_morphed_labels_fvs, sorted_genuine_labels_fvs, CURRENT_DB, CURRENT_METHOD)
        fvs_genuine = get_genuine_genuine_pairs(sorted_genuine_labels_fvs, sorted_genuine4morphed_labels_fvs, CURRENT_IMG_TYPE, CURRENT_DB, CURRENT_METHOD)
    else:
        fvs_morphed = process_pairs(get_morphed_genuine_pairs(morphed, genuine4morphed, CURRENT_DB, CURRENT_METHOD))
        fvs_genuine = process_pairs(get_genuine_genuine_pairs(genuine, genuine, CURRENT_IMG_TYPE, CURRENT_DB, CURRENT_METHOD))

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
            fv.append((apply_lbph_to_patched_img([morph, gen], 4), name))
        elif CURRENT_METHOD == PPC8_LBPH:
            fv.append((apply_lbph_to_patched_img([morph, gen], 8), name))
        elif CURRENT_METHOD == PPC12_LBPH:
            fv.append((apply_lbph_to_patched_img([morph, gen], 12), name))
        elif CURRENT_METHOD == PPC16_LBPH:
            fv.append((apply_lbph_to_patched_img([morph, gen], 16), name))
    return fv


def find_and_save_best_clf_for_current_method():
    # load_data_for_current_method(build_string_db_for_json(), CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)
    clf, feature_vectors, classes, ids = load_data_for_current_method(build_string_db_for_json(), CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)
    print("Grid search for method {} {} {}".format(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN))
    clf.grid_search_cross_val(feature_vectors, classes, CURRENT_TASK == TASK_TESTING)


def calculate_frr_intradb():
    clf, feature_vectors, classes, ids = load_data_for_current_method(build_string_db_for_json(), CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)
    print("\nFAR/FRR for method {} {} {}".format(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN))
    clf.get_frr(feature_vectors, classes, ids, True)


def calculate_frr_extradb(img_type_training=IMG_TYPE_DIGITAL, img_type_testing=IMG_TYPE_DIGITAL):
    clf, _, _, _ = load_data_for_current_method('../assets/db/' + TASK_TRAINING + "/" + img_type_training + "/" + TRAINING_DB + "/json/", CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)
    _, feature_vectors, classes, ids = load_data_for_current_method('../assets/db/' + TASK_TESTING + "/" + img_type_testing + "/" + TESTING_DB + "/json/", CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)
    print("\nFAR/FRR for method {} {} {}".format(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN))
    clf.get_frr(feature_vectors, classes, ids, False)


def calculate_fusion_frr(img_type_training, img_type_testing, fvc=FVC_LBPH, dfc=FVC_CNN, dim=DIM_96, align=ALIGN_EYES_LIP):
    print()
    print(fvc, dfc, dim, align, TRAINING_DB, TESTING_DB)

    is_intra = TRAINING_DB == TESTING_DB and img_type_training == img_type_testing

    if is_intra:
        clf_a, feature_vectors_a, classes_a, ids_a = load_data_for_current_method('../assets/db/' + TASK_TESTING + "/" + img_type_testing + "/" + TESTING_DB + "/json/", fvc, dim, align)
        clf_b, feature_vectors_b, classes_b, ids_b = load_data_for_current_method('../assets/db/' + TASK_TESTING + "/" + img_type_testing + "/" + TESTING_DB + "/json/", dfc, dim, align)
    else:
        clf_a, _, _, _ = load_data_for_current_method('../assets/db/' + TASK_TRAINING + "/" + img_type_training + "/" + TRAINING_DB + "/json/", fvc, dim, align)
        _, feature_vectors_a, classes_a, ids_a = load_data_for_current_method('../assets/db/' + TASK_TESTING + "/" + img_type_testing + "/" + TESTING_DB + "/json/", fvc, dim, align)
        clf_b, _, _, _ = load_data_for_current_method('../assets/db/' + TASK_TRAINING + "/" + img_type_training + "/" + TRAINING_DB + "/json/", dfc, dim, align)
        _, feature_vectors_b, classes_b, ids_b = load_data_for_current_method('../assets/db/' + TASK_TESTING + "/" + img_type_testing + "/" + TESTING_DB + "/json/", dfc, dim, align)

    clf_a = clf_a._load_classifier()
    clf_b = clf_b._load_classifier()

    min_fv_len = min(len(feature_vectors_a), len(feature_vectors_b))
    feature_vectors_a = feature_vectors_a[:min_fv_len]
    feature_vectors_b = feature_vectors_b[:min_fv_len]
    classes_a = classes_a[:min_fv_len]
    classes_b = classes_b[:min_fv_len]

    if is_intra:
        from sklearn.model_selection import train_test_split
        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(feature_vectors_a, classes_a, test_size=0.3, random_state=42)
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(feature_vectors_b, classes_b, test_size=0.3, random_state=42)
    else:
        X_test_a = feature_vectors_a
        X_test_b = feature_vectors_b
        y_test_a = classes_a
        y_test_b = classes_b

    y_probs_a = clf_a.predict_proba(X_test_a)
    y_probs_b = clf_b.predict_proba(X_test_b)

    # y_probs = np.mean([y_probs_a, y_probs_b], axis=0)
    y_probs = np.average([y_probs_a, y_probs_b], axis=0, weights=[2, 1])

    from SVM.metrics_utils import compute_frr_at_given_far_from_probabilities
    fars = [0.05, 0.1]
    frrs = []
    farsv = []
    for far in fars:
        frr, farv = compute_frr_at_given_far_from_probabilities(y_probs, far, y_test_a, 1)
        if frr is None or farv is None:
            frrs.append("100%")
            farsv.append(str(round(far * 100, 2)) + "%")
        else:
            frrs.append(str(round(frr * 100, 2)) + "%")
            farsv.append(str(round(farv * 100, 2)) + "%")
    header = ["5.00%", "10.00%"]
    rows = [frrs, farsv]
    from tabulate import tabulate
    print(tabulate(rows, headers=header, floatfmt=".2f"))


def run_all_fusion():
    # extractions = [[FVC_LBPH, FVC_CNN], [DFC_LBPH, DFC_CNN]]
    extractions = [[FVC_LBPH, FVC_CNN], [DFC_LBPH, DFC_CNN], [PPC8_LBPH, DFC_CNN], [PPC12_LBPH, DFC_CNN]]

    dimensions = [DIM_96, DIM_256]
    alignments = [ALIGN_DLIB, ALIGN_EYES_NOSE, ALIGN_EYES_LIP]
    for extr in extractions:
        for dim in dimensions:
            for align in alignments:
                if dim == DIM_96 and align == ALIGN_DLIB: continue
                calculate_fusion_frr(IMG_TYPE_DIGITAL, IMG_TYPE_PS, extr[0], extr[1], dim, align)


def run_all():
    # extractions = [PPC4_LBPH, PPC8_LBPH, PPC12_LBPH, PPC16_LBPH]
    extractions = [FVC_LBPH, DFC_LBPH, PPC8_LBPH, PPC12_LBPH, FVC_CNN, DFC_CNN]
    dimensions = [DIM_96, DIM_256]
    alignments = [ALIGN_DLIB, ALIGN_EYES_NOSE, ALIGN_EYES_LIP]

    global CURRENT_METHOD
    global CURRENT_ALIGN
    global CURRENT_DIM

    for method in extractions:
        CURRENT_METHOD = method
        for dimension in dimensions:
            CURRENT_DIM = dimension
            for alignment in alignments:
                CURRENT_ALIGN = alignment
                if (CURRENT_METHOD == FVC_CNN_OF or CURRENT_METHOD == DFC_CNN_OF) and \
                        ((CURRENT_DIM == DIM_96 and CURRENT_ALIGN == ALIGN_DLIB) or CURRENT_DIM == DIM_256): continue
                if CURRENT_DIM == DIM_96 and CURRENT_ALIGN == ALIGN_DLIB: continue
                do_stuff()


def do_stuff():
    print(CURRENT_METHOD, CURRENT_DIM, CURRENT_ALIGN)
    global morphed
    global genuine4morphed
    global genuine
    morphed, genuine4morphed, genuine = load_imgs_dbs(build_string_db())

    # feature_extraction_to_json()
    # find_and_save_best_clf_for_current_method()
    # calculate_frr_intradb()
    calculate_frr_extradb(IMG_TYPE_DIGITAL, IMG_TYPE_PS)


def build_string_db():
    return CURRENT_TASK + "/" + CURRENT_IMG_TYPE + "/" + CURRENT_DB + "/" + CURRENT_DIM + CURRENT_ALIGN


def build_string_db_for_json():
    return '../assets/db/' + CURRENT_TASK + "/" + CURRENT_IMG_TYPE + "/" + CURRENT_DB + "/json/"


def build_string_db_for_svm():
    return '../assets/db/' + CURRENT_TASK + "/" + CURRENT_IMG_TYPE + "/" + CURRENT_DB + "/svm/"


def build_string_db_for_openface_labels():
    return '../assets/db/' + CURRENT_TASK + "/" + CURRENT_IMG_TYPE + "/" + CURRENT_DB + "/" + CURRENT_DIM + CURRENT_ALIGN + "/"


if __name__ == '__main__':
    TRAINING_DB = DB_PMDB
    TESTING_DB = DB_MORPHEDDB
    CURRENT_TASK = TASK_TESTING

    CURRENT_DB = TESTING_DB
    CURRENT_IMG_TYPE = IMG_TYPE_DIGITAL
    CURRENT_DIM = DIM_256
    CURRENT_ALIGN = ALIGN_DLIB

    CURRENT_METHOD = DFC_CNN

    morphed, genuine4morphed, genuine = load_imgs_dbs(build_string_db())

    morphed_labels, morphed_fv, sorted_morphed_labels_fvs, \
    genuine4morphed_labels, genuine4morphed_fv, sorted_genuine4morphed_labels_fvs, \
    genuine_labels, genuine_fv, sorted_genuine_labels_fvs = load_imgs_labels_and_csv(CURRENT_METHOD, build_string_db_for_openface_labels())

    # save_preprocessed_originals_to_file()

    # feature_extraction_to_json()
    # find_and_save_best_clf_for_current_method()
    # calculate_frr_intradb()
    # calculate_frr_extradb(IMG_TYPE_PS, IMG_TYPE_PS)

    run_all()
    run_all_fusion()

    pass
