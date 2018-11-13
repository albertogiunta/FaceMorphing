import csv
import glob
import json

ALIGN_NONE = ""
DIM_ORIGINAL = "original"


def _load_csv(filename, filetype):
    fv = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if filetype == "labels":
                fv.append(row[1])
            elif filetype == "reps":
                fv.append([float(i) for i in row])
    return fv


def load_imgs_labels_and_csv(current_dim, current_align):
    if current_dim != "96":
        return [], [], [], [], [], [], [], [], []

    db_category = current_dim + current_align
    # biometix_morphed_labels = _load_csv("../assets/db/" + db_category + "/" + "biometix-morphed-csv-rep/labels.csv", "labels")
    # biometix_morphed_fv = _load_csv("../assets/db/" + db_category + "/" + "biometix-morphed-csv-rep/reps.csv", "reps")
    # sorted_biometix_morphed_labels_fvs = _join_and_sort(biometix_morphed_labels, biometix_morphed_fv)
    # biometix_genuine_labels = _load_csv("../assets/db/" + db_category + "/" + "biometix-genuine-csv-rep/labels.csv", "labels")
    # biometix_genuine_fv = _load_csv("../assets/db/" + db_category + "/" + "biometix-genuine-csv-rep/reps.csv", "reps")
    # sorted_biometix_genuine_labels_fvs = _join_and_sort(biometix_genuine_labels, biometix_genuine_fv)
    # feret_genuine_labels = _load_csv("../assets/db/" + db_category + "/" + "feret-genuine-csv-rep/labels.csv", "labels")
    # feret_genuine_fv = _load_csv("../assets/db/" + db_category + "/" + "feret-genuine-csv-rep/reps.csv", "reps")
    # sorted_feret_genuine_labels_fvs = _join_and_sort(feret_genuine_labels, feret_genuine_fv)

    morphed_labels = _load_csv("../assets/db/digital/" + db_category + "/" + "morphed-csv-rep/labels.csv", "labels")
    morphed_fv = _load_csv("../assets/db/digital/" + db_category + "/" + "morphed-csv-rep/reps.csv", "reps")
    sorted_morphed_labels_fvs = _join_and_sort(morphed_labels, morphed_fv)
    genuine4morphed_labels = _load_csv("../assets/db/digital/" + db_category + "/" + "genuine4morphed-csv-rep/labels.csv", "labels")
    genuine4morphed_fv = _load_csv("../assets/db/digital/" + db_category + "/" + "genuine4morphed-csv-rep/reps.csv", "reps")
    sorted_genuine4morphed_labels_fvs = _join_and_sort(genuine4morphed_labels, genuine4morphed_fv)
    genuine_labels = _load_csv("../assets/db/digital/" + db_category + "/" + "genuine-csv-rep/labels.csv", "labels")
    genuine_fv = _load_csv("../assets/db/digital/" + db_category + "/" + "genuine-csv-rep/reps.csv", "reps")
    sorted_genuine_labels_fvs = _join_and_sort(genuine_labels, genuine_fv)

    return morphed_labels, morphed_fv, sorted_morphed_labels_fvs, \
           genuine4morphed_labels, genuine4morphed_fv, sorted_genuine4morphed_labels_fvs, \
           genuine_labels, genuine_fv, sorted_genuine_labels_fvs


def load_imgs_dbs(current_dim, current_align):
    db_category = current_dim + current_align
    # biometix_morphed = sorted(glob.glob("../assets/db/" + db_category + "/" + "biometix-morphed/imgs/*.*"))
    # biometix_genuine = sorted(glob.glob("../assets/db/" + db_category + "/" + "biometix-genuine/imgs/*.*"))
    # feret_genuine = sorted(glob.glob("../assets/db/" + db_category + "/" + "feret-genuine/imgs/*.*"))
    morphed = sorted(glob.glob("../assets/db/digital/" + db_category + "/" + "morphed/imgs/*.*"))
    genuine4morphed = sorted(glob.glob("../assets/db/digital/" + db_category + "/" + "genuine4morphed/imgs/*.*"))
    genuine = sorted(glob.glob("../assets/db/digital/" + db_category + "/" + "genuine/imgs/*.*"))

    return morphed, genuine4morphed, genuine


def load_data_for_current_method(current_method, current_dim, current_align):
    print("Loading data from file")
    with open('../assets/json/' + current_method + "_" + current_dim + "_" + current_align + '.json', 'r') as infile:
        data = json.load(infile)
        print("\tLoaded")

        feature_vectors = []
        classes = []

        if len(data) == 2:
            for cls in data:
                for el in cls:
                    if el['fv'] is not None:
                        feature_vectors.append(el['fv'])
                        classes.append(el['cls'])
        else:
            for el in data:
                if el['fv'] is not None:
                    feature_vectors.append(el['fv'])
                    classes.append(el['cls'])

        print("\tData structures created - {} samples loaded".format(len(feature_vectors)))

        return get_svm_classifier(current_method, current_dim, current_align), feature_vectors, classes


def get_svm_classifier(current_method, current_dim, current_align):
    from SVM.svm_classification import SVMClassifier
    svm_classifier = SVMClassifier(current_method, current_dim, current_align)
    return svm_classifier


def get_data_to_be_written(feature_vectors, cls):
    result = []
    for fv in feature_vectors:
        data = {'fv': fv, 'cls': cls}
        result.append(data)
    return result


def save_to_file(data, current_method, current_dim, current_align):
    from utils.config_utils import NumpyEncoder
    with open('../assets/json/' + current_method + "_" + current_dim + "_" + current_align + '.json', 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)


def _join_and_sort(a, b):
    result = []
    if len(a) == len(b):
        for i in range(len(a)):
            result.append([a[i], b[i]])

    return sorted(result, key=lambda x: x[0])
