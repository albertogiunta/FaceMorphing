# https://github.com/davisking/dlib/blob/master/python_examples/svm_binary_classifier.py

import pickle

from sklearn import svm


class SVMClassifier:

    def __init__(self):
        self.svm = svm.SVC
        self.classifier = self.svm(kernel='rbf')

    # For binary classification, the classes labels should all be either +1 or -1.

    # model_name can be:

    # FVC_CNN, DFC_CNN, MSPPC_CNN
    # PPC4_CNN, PPC8_CNN, PPC12_CNN, PPC16_CNN

    # FVC_LBPH, DFC_LBPH, MSPPC_LBPH
    # PPC4_LBPH, PPC8_LBPH, PPC12_LBPH, PPC16_LBPH

    def train(self, feature_vectors, classes, model_name):
        self.classifier = svm.fit(feature_vectors, classes)

        with open('../models/SVM_model_' + model_name + '.pickle', 'wb') as handle:
            pickle.dump(self.classifier, handle, 2)

    def predict_class(self, feature_vector):
        prediction_value = self.classifier.predict(feature_vector)
        print("Prediction value: {}".format(prediction_value))
        return prediction_value

# if __name__ == '__main__':
#     feature_vectors = dlib.vectors()
#     classes = dlib.array()
#
#     feature_vectors.append(dlib.vector([1, 2, 3, -1, -2, -3]))
#     feature_vectors.append(dlib.vector([-1, -2, -3, 1, 2, 3]))
#     classes.append(+1)
#     classes.append(-1)
#
#     svm = SVMClassifier()
#     svm.train(feature_vectors, classes)
#     svm.predict_class(feature_vectors[0])
