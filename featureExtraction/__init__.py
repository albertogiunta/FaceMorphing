from featureExtraction.fve_cnn import CNNFeatureVectorExtraction
from utils import models_utils

if __name__ == '__main__':
    shape_predictor_path, face_rec_model_path, faces_pairs_path = models_utils.model_paths()

    face_rec = CNNFeatureVectorExtraction(shape_predictor_path, face_rec_model_path)
    feature_vectors = face_rec.get_img_pair_descriptors(faces_pairs_path)
    euclidean_distance = face_rec.calculate_euclidean_distance(feature_vectors[0], feature_vectors[1])

    print("Match found: {}".format(face_rec.is_pair_from_same_person(euclidean_distance)))
    print("Euclidean distance: {}".format(euclidean_distance))
