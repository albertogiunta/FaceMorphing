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


def load_imgs_labels_and_csv(current_dim, current_align, current_db):
    if current_dim != "96":
        return [], [], [], [], [], [], [], [], []

    db_category = current_dim + current_align
    morphed_labels = _load_csv("../assets/db/" + current_db + "/" + db_category + "/" + "morphed-csv-rep/labels.csv", "labels")
    morphed_fv = _load_csv("../assets/db/" + current_db + "/" + db_category + "/" + "morphed-csv-rep/reps.csv", "reps")
    sorted_morphed_labels_fvs = _join_and_sort(morphed_labels, morphed_fv)
    genuine4morphed_labels = _load_csv("../assets/db/" + current_db + "/" + db_category + "/" + "genuine4morphed-csv-rep/labels.csv", "labels")
    genuine4morphed_fv = _load_csv("../assets/db/" + current_db + "/" + db_category + "/" + "genuine4morphed-csv-rep/reps.csv", "reps")
    sorted_genuine4morphed_labels_fvs = _join_and_sort(genuine4morphed_labels, genuine4morphed_fv)
    genuine_labels = _load_csv("../assets/db/" + current_db + "/" + db_category + "/" + "genuine-csv-rep/labels.csv", "labels")
    genuine_fv = _load_csv("../assets/db/" + current_db + "/" + db_category + "/" + "genuine-csv-rep/reps.csv", "reps")
    sorted_genuine_labels_fvs = _join_and_sort(genuine_labels, genuine_fv)

    return morphed_labels, morphed_fv, sorted_morphed_labels_fvs, \
           genuine4morphed_labels, genuine4morphed_fv, sorted_genuine4morphed_labels_fvs, \
           genuine_labels, genuine_fv, sorted_genuine_labels_fvs


def load_imgs_dbs(current_dim, current_align, current_db):
    db_category = current_dim + current_align
    morphed = sorted(glob.glob("../assets/db/" + current_db + "/" + db_category + "/" + "morphed/imgs/*.*"))
    genuine4morphed = sorted(glob.glob("../assets/db/" + current_db + "/" + db_category + "/" + "genuine4morphed/imgs/*.*"))
    genuine = sorted(glob.glob("../assets/db/" + current_db + "/" + db_category + "/" + "genuine/imgs/*.*"))

    return morphed, genuine4morphed, genuine


def load_data_for_current_method(current_db, current_method, current_dim, current_align, tot_to_be_loaded):
    # print("Loading data from file")
    with open('../assets/db/' + current_db + '/json/' + current_method + "_" + current_dim + "_" + current_align + '.json', 'r') as infile:
        data = json.load(infile)
        # print("\tLoaded")

        feature_vectors = []
        classes = []
        ids = []

        if len(data) == 2:
            for cls in data:
                for el in cls:
                    if el['fv'] is not None:
                        feature_vectors.append(el['fv'])
                        classes.append(el['cls'])
                        ids.append(el['id'])
        else:
            print("PROBLEM WHILE LOADING DATA")
        #     for el in data:
        #         if el['fv'] is not None:
        #             feature_vectors.append(el['fv'])
        #             classes.append(el['cls'])

        # print("\tData structures created - {} samples loaded".format(len(feature_vectors)))

        return get_svm_classifier(current_db, current_method, current_dim, current_align), feature_vectors[:tot_to_be_loaded], classes[:tot_to_be_loaded], ids[:tot_to_be_loaded]


def get_svm_classifier(current_db, current_method, current_dim, current_align):
    from SVM.svm_classification import SVMClassifier
    svm_classifier = SVMClassifier(current_db, current_method, current_dim, current_align)
    return svm_classifier


def get_data_to_be_written(feature_vectors, cls):
    result = []
    for el in feature_vectors:
        fv = el[0]
        id = el[1]
        data = {'fv': fv, 'cls': cls, 'id': id}
        result.append(data)
    return result


def save_to_file(data, current_db, current_method, current_dim, current_align):
    from utils.config_utils import NumpyEncoder
    with open('../assets/db/' + current_db + '/json/' + current_method + "_" + current_dim + "_" + current_align + '.json', 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)


def _join_and_sort(a, b):
    result = []
    if len(a) == len(b):
        for i in range(len(a)):
            result.append([a[i], b[i]])

    return sorted(result, key=lambda x: x[0])
