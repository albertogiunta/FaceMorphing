from featureExtraction.fve_cnn import CNNFeatureVectorExtraction
from utils import models_utils

if __name__ == '__main__':
    shape_predictor_path, face_rec_model_path, faces_pairs_path = models_utils.model_paths()

    face_rec = CNNFeatureVectorExtraction(shape_predictor_path, face_rec_model_path)
    feature_vectors = face_rec.get_img_pair_descriptors(faces_pairs_path)
