# https://github.com/davisking/dlib/blob/master/python_examples/svm_binary_classifier.py
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tabulate import tabulate

class SVMClassifier:

    def __init__(self, db_string, svm_basename):
        self.clf = None
        self.db_string = db_string
        self.svm_basename = svm_basename
        self.model_name = svm_basename

    def grid_search_cross_val(self, feature_vectors, classes, is_intra):
        if is_intra:
            from sklearn.model_selection import train_test_split
            X_train, _, y_train, _ = train_test_split(feature_vectors, classes, test_size=0.3, random_state=42)
        else:
            X_train = feature_vectors
            y_train = classes

        tuned_parameters = [
            # {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'max_iter': [2000]}
        ]

        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, n_jobs=-1)
        clf.fit(X_train, y_train)

        print("Best parameters set found:")
        print(clf.best_params_)
        self.model_name = self.svm_basename + "_" + str(clf.best_params_['C']) + "_" + clf.best_params_['kernel']
        self.clf = clf.best_estimator_
        self._save_classifier_to_disk()

    def get_frr(self, feature_vectors, classes, ids, is_intra):
        from SVM.metrics_utils import compute_frr_at_given_far_from_probabilities
        self._load_classifier()

        fars = [0.05, 0.1]
        frrs = []
        farsv = []

        if is_intra:
            _, X_test, _, y_test, _, ids_test = train_test_split(feature_vectors, classes, ids, test_size=0.3, random_state=42)
        else:
            X_test = feature_vectors
            y_test = classes
            ids_test = ids

        y_probs = self.clf.predict_proba(X_test)

        print("Score: {}".format(self.clf.score(X_test, y_test)))

        for far in fars:
            frr, far_v = compute_frr_at_given_far_from_probabilities(y_probs, far, y_test, 1)
            if frr is None and far_v is None:
                print("FRR IS NONE")
                continue
            frrs.append(str(round(frr * 100, 2)) + "%")
            farsv.append(str(round(far_v * 100, 2)) + "%")

        self._print_apcer_bpcer_table(frrs, farsv)
        self._print_conf_matrix(y_test, self.clf.predict(X_test))
        return

    def _print_apcer_bpcer_table(self, bpcer, apcer):
        header = ["5.00%", "10.00%"]
        rows = [bpcer, apcer]
        print(tabulate(rows, headers=header, floatfmt=".2f"))

    def _save_classifier_to_disk(self):
        from launcher.data_io import save_classifier
        save_classifier(self.clf, self.db_string, self.model_name)

    def _load_classifier(self):
        from launcher.data_io import load_classifier
        self.clf = load_classifier(self.clf, self.db_string, self.model_name)
        return self.clf

    def _print_conf_matrix(self, y_test, y_pred):
        print()
        print(confusion_matrix(y_test, y_pred))

    def _print_classification_report(self, y_test, y_pred):
        print()
        print(classification_report(y_test, y_pred, target_names=['morphed', 'genuine']))
