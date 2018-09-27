import os

from preprocessing.preprocessing import Preprocessing
from utils import img_utils

if __name__ == '__main__':
    pair_path = os.path.join("../img/pair1", "*." + "png")
    plain_imgs = img_utils.load_imgs_in_folder(pair_path)

    # preprocess
    preproc_imgs = []
    preproc = Preprocessing()

    for img in plain_imgs:
        preproc_img = preproc.preprocess_img(img)
        preproc_imgs.append(preproc_img)

    # feature_vectors = []
    #
    # lbph = LBPHFeatureVectorExtraction()
    # for img in preproc_imgs:
    #     feature_vector = lbph.get_img_descriptor_from_path(img)
    #     feature_vectors.append(feature_vector)

    # face_rec = CNNFeatureVectorExtraction()
    # feature_vectors = face_rec.get_img_pair_descriptors(pair_path)
