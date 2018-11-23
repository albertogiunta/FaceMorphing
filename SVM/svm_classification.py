# https://github.com/davisking/dlib/blob/master/python_examples/svm_binary_classifier.py
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tabulate import tabulate


class SVMClassifier:

    def __init__(self, current_db, model_name, current_dim, current_align):
        self.clf = None
        self.current_db = current_db
        self.model_name = model_name
        self.current_dim = current_dim
        self.current_align = current_align

    def grid_search_cross_val(self, feature_vectors, classes):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(feature_vectors, classes, test_size=0.3, random_state=42)

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
        self.model_name = self.model_name + "_" + self.current_dim + "_" + self.current_align + "_" + \
                          str(clf.best_params_['C']) + "_" + clf.best_params_['kernel']
        self.clf = clf.best_estimator_
        self._save_classifier_to_disk()
        y_pred = self.clf.predict(X_test)
        print("Accuracy: {}".format(self.clf.score(X_test, y_test)))
        self._print_conf_matrix(y_test, y_pred)
        self._print_classification_report(y_test, y_pred)

    def get_frr(self, feature_vectors, classes, ids):
        from SVM.metrics_utils import compute_frr_at_given_far_from_probabilities

        self._load_classifier()

        fars = [0.05, 0.1]
        frrs = []
        farsv = []

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(feature_vectors, classes, ids, test_size=0.3, random_state=42)
        y_probs = self.clf.predict_proba(X_test)

        print("Score: {}".format(self.clf.score(X_test, y_test)))

        for far in fars:
            frr, far_v = compute_frr_at_given_far_from_probabilities(y_probs, far, y_test, 1)
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
        save_classifier(self.clf, self.current_db, self.model_name)

    def _load_classifier(self):
        from launcher.data_io import load_classifier
        load_classifier(self.clf, self.current_db, self.model_name, self.current_dim, self.current_align)

    def _print_conf_matrix(self, y_test, y_pred):
        print()
        print(confusion_matrix(y_test, y_pred))

    def _print_classification_report(self, y_test, y_pred):
        print()
        print(classification_report(y_test, y_pred, target_names=['morphed', 'genuine']))
