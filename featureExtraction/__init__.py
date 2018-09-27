from featureExtraction.cnn_fv_extraction import CNNFeatureVectorExtraction

if __name__ == '__main__':
    face_rec = CNNFeatureVectorExtraction()
    feature_vectors = face_rec.get_img_pair_descriptors("../img")
    # euclidean_distance = face_rec.calculate_euclidean_distance(feature_vectors[0], feature_vectors[1])

    # print("Match found: {}".format(face_rec.is_pair_from_same_person(euclidean_distance)))
    # print("Euclidean distance: {}".format(euclidean_distance))
