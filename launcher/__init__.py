import glob
import re

import dlib
from skimage import io

from SVM.svm_classification import SVMClassifier
from featureExtraction.lbph_fv_extraction import LBPHFeatureVectorExtraction
from featureVectorMerging.differential_comparison import DifferentialComparison
from preprocessing.preprocessing import Preprocessing

if __name__ == '__main__':

    imgs_to_be_checked = 1
    pattern_extract_genuine = re.compile('([0-9]{5})(?!\_)')
    pattern_extract_genuine_ids_from_morphed = re.compile(r'([0-9]{5})')

    all_morphed = glob.glob("../biometrix/morphed/*.jpg")
    all_genuine = glob.glob("../biometrix/genuine/*.png")

    for i, img in enumerate(all_morphed):
        if i >= imgs_to_be_checked:
            break

        morphed_path = img

        genuine_ids = re.findall(pattern_extract_genuine_ids_from_morphed, morphed_path)
        base_genuine_id = genuine_ids[1]  # change to 0 to check the other image used for the attack
        pattern_match_genuine = re.compile('(.+' + base_genuine_id + '.+)')
        genuine_path = next(x for x in all_genuine if pattern_match_genuine.search(x) is not None)

        print("Processing images:\n"
              "Genuine (Base) : {}\n"
              "Morphed: {} + {}\n"
              .format(base_genuine_id, genuine_ids[0], genuine_ids[1]))

        # preprocess
        preproc_imgs = []  # 1. genuine 2. morphed
        preproc = Preprocessing()

        preproc_imgs.append(preproc.preprocess_img_path(genuine_path))
        preproc_imgs.append(preproc.preprocess_img_path(morphed_path))

        io.imshow_collection(preproc_imgs)
        io.show()

        # lbph
        feature_vectors = []
        lbph = LBPHFeatureVectorExtraction()
        feature_vectors.append(lbph.get_img_descriptor_from_img(preproc_imgs[0]))
        feature_vectors.append(lbph.get_img_descriptor_from_img(preproc_imgs[1]))
        differential = DifferentialComparison.get_differential_fv(feature_vectors[1], feature_vectors[0])

        # SVM
        dlib_feature_vectors = dlib.vectors()
        classes = dlib.array()

        dlib_feature_vectors.append(dlib.vector(feature_vectors[0]))
        dlib_feature_vectors.append(dlib.vector(feature_vectors[1]))
        classes.append(+1)
        classes.append(-1)

        svm = SVMClassifier()
        svm.train(dlib_feature_vectors, classes)
        svm.predict_class(dlib_feature_vectors[0])
