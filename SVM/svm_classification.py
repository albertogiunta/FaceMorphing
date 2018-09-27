# https://github.com/davisking/dlib/blob/master/python_examples/svm_binary_classifier.py

import dlib

try:
    import cPickle as pickle
except ImportError:
    import pickle


class SVMClassifier:

    def __init__(self):
        self.svm = dlib.svm_c_trainer_radial_basis()
        # self.svm.set_c(10)
        self.classifier = None

    def train(self, feature_vectors, classes):
        # For binary classification, the classes labels should all be either +1 or -1.

        self.classifier = self.svm.train(feature_vectors, classes)

        with open('../models/SVM_classifier_model.pickle', 'wb') as handle:
            pickle.dump(self.classifier, handle, 2)

    def predict_class(self, feature_vector):
        if self.classifier is None:
            print("SVM Classifier was not trained.")
            exit(0)

        prediction_value = self.classifier(feature_vector)

        print("Prediction value: {}".format(prediction_value))

        return prediction_value


if __name__ == '__main__':
    feature_vectors = dlib.vectors()
    classes = dlib.array()

    feature_vectors.append(dlib.vector([1, 2, 3, -1, -2, -3]))
    feature_vectors.append(dlib.vector([-1, -2, -3, 1, 2, 3]))
    classes.append(+1)
    classes.append(-1)

    svm = SVMClassifier()
    svm.train(feature_vectors, classes)
    svm.predict_class(feature_vectors[0])
