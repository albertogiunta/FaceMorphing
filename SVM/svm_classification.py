# https://github.com/davisking/dlib/blob/master/python_examples/svm_binary_classifier.py
import pickle
import sys

import numpy as np
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tabulate import tabulate


class SVMClassifier:

    def __init__(self, model_name, current_dim, current_align):
        self.clf = None
        self.model_name = model_name
        self.current_dim = current_dim
        self.current_align = current_align

    def grid_search_cross_val(self, feature_vectors, classes):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(feature_vectors, classes, test_size=0.3, random_state=42)

        tuned_parameters = [
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'max_iter': [2000]}
            # {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [3, 4, 5], 'gamma': [1e-3, 1e-4]}
        ]

        scores = ['recall']

        for score in scores:
            from sklearn.model_selection import GridSearchCV
            from sklearn.svm import SVC
            clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='%s_macro' % score)
            clf.fit(X_train, y_train)

            print("Best parameters set found:")
            print(clf.best_params_)
            print()
            self.model_name = self.model_name + "_" + self.current_dim + "_" + self.current_align + "_" + \
                              str(clf.best_params_['C']) + "_" + clf.best_params_['kernel']
            self.clf = clf.best_estimator_
            self._save_classifier()
            y_pred = self.clf.predict(X_test)
            print("Accuracy: {}".format(self.clf.score(X_test, y_test)))
            print()
            print(confusion_matrix(y_test, y_pred))
            print()
            print(classification_report(y_test, y_pred, target_names=['morphed', 'genuine']))

    def get_bpcer(self, feature_vectors, classes):
        # apcer - attack presentation classification error rate: proportion of morphed presentations incorrectly classified as bona fide
        # bpcer - bonafide presentation classification error rate: proportion of bona fide presentations incorrectly classified as morphed face attacks
        apcer = [0.05, 0.1]
        bpcer = []

        X_train, X_test, y_train, y_test = train_test_split(feature_vectors, classes, test_size=0.3, random_state=42)
        # X_test = feature_vectors
        # y_test = classes
        self._load_classifier()

        # print(self.clf.score(X_test, y_test))

        # prediction probability for every test element for both -1 and 1 class
        y_scores = self.clf.predict_proba(X_test)[:, 1]
        # decision function scores for every test element
        # y_scores2 = self.clf.decision_function(X_test)

        # precision and recall values at different threshold values
        precisions, _, thresholds = precision_recall_curve(y_test, y_scores)
        # total number of morphed images in test group
        num_morphed = len([el for el in y_test if (el == -1 or el == 0)])

        # plt.plot(precisions)
        # plt.show()
        # plt.plot(thresholds)
        # plt.show()

        for apcer_rate in apcer:
            # number of accepted false positives at this apcer_rate
            num_false_positives = int(num_morphed * apcer_rate)

            # precision wanted at this level of apcer_rate.
            # precision formula was used: tp / (tp + fp)
            # where tp = num_morphed, fp = num_false_positives
            current_precision = num_morphed / (num_morphed + num_false_positives)

            # the precision closest to the one we've found in the precision array
            # needed in order to find the corresponding threshold
            approx_precision = self.find_nearest(precisions, current_precision)
            approx_precision_index = precisions.tolist().index(approx_precision)

            # minimum threshold corresponding to the wanted morphed fp precision
            min_threshold = thresholds[approx_precision_index] + sys.float_info.epsilon

            # array of predictions adjusted on the newly found minimum threshold
            y_pred_adjusted = [1 if y >= min_threshold else 0 for y in y_scores]

            ############################################################################################################
            conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
            current_bpcer = str(round(conf_matrix[1][0] * 100 / (conf_matrix[1][1] + conf_matrix[1][0]), 2)) + "%"
            bpcer.append(current_bpcer)

            # print()
            # print("APCER: {}%  --- BPCER: {}".format(apcer_rate*100, current_bpcer))
            # print(current_precision, approx_precision)
            # print(pd.DataFrame(conf_matrix, columns=['pred_neg', 'pred_pos'], index=['morphed', 'genuine']))
            # print(classification_report(y_test, y_pred_adjusted, target_names=['morphed', 'genuine']))

        print()
        self._print_apcer_bpcer_table(bpcer)

    def get_frr(self, feature_vectors, classes):
        from SVM.metrics_utils import compute_frr_at_given_far_from_probabilities

        self._load_classifier()
        classes = [0 if el == -1 else el for el in classes]


        fars = [0.05, 0.1]
        frrs = []

        X_train, X_test, y_train, y_test = train_test_split(feature_vectors, classes, test_size=0.3, random_state=42)
        y_probs = self.clf.predict_proba(X_test)

        print("Score: {}".format(self.clf.score(X_test, y_test)))

        for far in fars:
            frr, _ = compute_frr_at_given_far_from_probabilities(y_probs, far, y_test, 1)
            frrs.append(str(round(frr * 100, 2)) + "%")

        self._print_apcer_bpcer_table(frrs)
        return

    def find_false_negatives(self, feature_vectors, classes):
        X_train, X_test, y_train, y_test = train_test_split(feature_vectors, classes, test_size=0.3, random_state=42)
        self._load_classifier()
        from launcher import biometix_runner
        from utils import img_utils
        num_genuine_train = classes.count(1)
        for i in range(len(X_test)):
            curr_fv = X_test[i]
            curr_cls = y_test[i]
            curr_index = feature_vectors.index(curr_fv)

            if curr_cls == -1 or curr_cls == 0:
                curr_index -= num_genuine_train
                ok_pred = self.clf.predict([curr_fv]) == curr_cls
                if not ok_pred:
                    # img_utils.show_img_skimage(biometix_runner.get_nth_morphed_image(curr_index))
                    morphed, genuine = biometix_runner.get_nth_morphed_genuine_pair(curr_index)
                    img_utils.show_imgs_skimage(
                        (img_utils.load_img_skimage(morphed), img_utils.load_img_skimage(genuine)))
                    # img_utils.show_img_skimage(img_utils.load_img_skimage(genuine))

    def find_nearest(self, array, value):
        new_array = []
        for i, el in enumerate(array[:-1]):
            if array[i] <= value:
                new_array.append(array[i])
        if len(new_array) == 0:
            new_array.append(array[0])
        new_array = np.asarray(new_array)
        idx = (np.abs(new_array - value)).argmin()
        return array[idx]

    def adjusted_classes(self, y_scores, t):
        return [1 if y >= t else 0 for y in y_scores]

    def _print_apcer_bpcer_table(self, bpcer):
        header = ["5.00%", "10.00%"]
        rows = [bpcer]
        print(tabulate(rows, headers=header, floatfmt=".2f"))

    def _save_classifier(self):
        with open('../assets/svm/' + self.model_name + '.pickle', 'wb') as handle:
            pickle.dump(self.clf, handle, 2)

    def _load_classifier(self):
        if self.clf is None:
            import glob
            self.clf = pickle.load(open(glob.glob(
                "../assets/svm/" + self.model_name + "_" + self.current_dim + "_" + self.current_align + "_" + "*.pickle")[
                                            0], 'rb'))
