import csv
import glob
import json
import os
import re


def load_imgs_labels_and_csv(current_method, db_string):
    DFC_CNN_OF = "DFC_CNN_OF"
    FVC_CNN_OF = "FVC_CNN_OF"
    if current_method != DFC_CNN_OF and current_method != FVC_CNN_OF:
        return [], [], [], [], [], [], [], [], []

    morphed_labels = _load_csv(db_string + "mor-csv-rep/labels.csv", "labels")
    morphed_fv = _load_csv(db_string + "mor-csv-rep/reps.csv", "reps")
    sorted_morphed_labels_fvs = _join_and_sort(morphed_labels, morphed_fv)
    genuine4morphed_labels = _load_csv(db_string + "g4m-csv-rep/labels.csv", "labels")
    genuine4morphed_fv = _load_csv(db_string + "g4m-csv-rep/reps.csv", "reps")
    sorted_genuine4morphed_labels_fvs = _join_and_sort(genuine4morphed_labels, genuine4morphed_fv)
    genuine_labels = _load_csv(db_string + "gen-csv-rep/labels.csv", "labels")
    genuine_fv = _load_csv(db_string + "gen-csv-rep/reps.csv", "reps")
    sorted_genuine_labels_fvs = _join_and_sort(genuine_labels, genuine_fv)

    return morphed_labels, morphed_fv, sorted_morphed_labels_fvs, \
           genuine4morphed_labels, genuine4morphed_fv, sorted_genuine4morphed_labels_fvs, \
           genuine_labels, genuine_fv, sorted_genuine_labels_fvs


def load_imgs_dbs(db_string):
    morphed = sorted(glob.glob("../assets/db/" + db_string + "/mor/imgs/*.*"))
    genuine4morphed = sorted(glob.glob("../assets/db/" + db_string + "/g4m/imgs/*.*"))
    genuine = sorted(glob.glob("../assets/db/" + db_string + "/gen/imgs/*.*"))
    return morphed, genuine4morphed, genuine


def load_data_for_current_method(db_string, current_method, current_dim, current_align):
    if len(re.compile("both").findall(db_string)) == 1:
        feature_vectors_dig, classes_dig, ids_dig = get_json_stuff(db_string.replace("json", "json_dig") + current_method + "_" + current_dim + current_align + '.json')
        feature_vectors_ps, classes_ps, ids_ps = get_json_stuff(db_string.replace("json", "json_ps") + current_method + "_" + current_dim + current_align + '.json')
        feature_vectors = feature_vectors_dig + feature_vectors_ps
        classes = classes_dig + classes_ps
        ids = ids_dig + ids_ps
    else:
        feature_vectors, classes, ids = get_json_stuff(db_string + current_method + "_" + current_dim + current_align + '.json')
    print("\t{} samples loaded into memory".format(len(feature_vectors)))
    return get_svm_classifier(db_string.replace("json", "svm"), "{}_{}".format(current_method, current_dim + current_align)), feature_vectors, classes, ids


def get_json_stuff(json_string):
    with open(json_string, 'r') as infile:
        data = json.load(infile)
        if len(data) >= 1000:
            data2 = []
            data2.append([x for x in data if x['cls'] == 0])
            data2.append([x for x in data if x['cls'] == 1])
            data = data2
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

            return (feature_vectors, classes, ids)

        else:
            print("PROBLEM WHILE LOADING DATA {}".format(len(data)))
            exit(0)


def get_svm_classifier(db_string, svm_basename):
    from SVM.svm_classification import SVMClassifier
    svm_classifier = SVMClassifier(db_string, svm_basename)
    return svm_classifier


def get_data_to_be_written(feature_vectors, cls):
    result = []
    for el in feature_vectors:
        fv = el[0]
        id = el[1]
        data = {'fv': fv, 'cls': cls, 'id': id}
        result.append(data)
    return result


def save_to_file(data, db_string, current_method, current_dim, current_align):
    from utils.config_utils import NumpyEncoder
    if not os.path.exists(db_string):
        os.makedirs(db_string)
    with open(db_string + current_method + "_" + current_dim + current_align + '.json', 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)


def save_classifier(clf, db_string, model_name):
    if not os.path.exists(db_string):
        os.makedirs(db_string)
    with open(db_string + model_name + '.pickle', 'wb') as handle:
        import pickle
        pickle.dump(clf, handle, 2)


def load_classifier(clf, db_string, model_name):
    print(db_string + model_name)
    if clf is None:
        import glob
        import pickle
        clf = pickle.load(open(glob.glob(db_string + model_name + "_*.pickle")[0], 'rb'))
    return clf


def _join_and_sort(a, b):
    result = []
    if len(a) == len(b):
        for i in range(len(a)):
            result.append([a[i], b[i]])
    return sorted(result, key=lambda x: x[0])


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
