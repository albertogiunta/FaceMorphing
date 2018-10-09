# https://github.com/davisking/dlib/blob/master/python_examples/svm_binary_classifier.py

import pickle

from sklearn import svm


class SVMClassifier:

    def __init__(self, model_name):
        # self.svm = svm.SVC(kernel='rbf')
        self.svm = svm.LinearSVC()
        self.classifier = None
        self.model_name = model_name

    def train(self, feature_vectors, classes):
        self.classifier = self.svm.fit(feature_vectors, classes)

        with open('../assets/svm/' + self.model_name + '.pickle', 'wb') as handle:
            pickle.dump(self.classifier, handle, 2)

    def predict_class(self, feature_vector):
        if self.classifier is None:
            self.classifier = pickle.load(open('../assets/svm/' + self.model_name + '.pickle', 'rb'))

        prediction_value = self.classifier.predict([feature_vector])
        print("Prediction value: {}".format(prediction_value))
        return prediction_value

    def get_score(self, feature_vectors, classes):
        if self.classifier is None:
            self.classifier = pickle.load(open('../assets/svm/' + self.model_name + '.pickle', 'rb'))

        return self.classifier.score(feature_vectors, classes)

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
