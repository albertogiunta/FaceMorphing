def apply_preproc(*args):
    from preprocessing.preprocessing import Preprocessing
    preproc = Preprocessing()
    preproc_imgs = []

    for img in args:
        new_img = preproc.preprocess_img(img)
        if new_img is not None:
            preproc_imgs.append(new_img)
    return preproc_imgs


def apply_cnn(*args):
    from featureExtraction.cnn_fv_extraction import CNNFeatureVectorExtraction
    cnn = CNNFeatureVectorExtraction()
    feature_vectors = []

    for img in args:
        feature_vectors.append(cnn.get_img_descriptor_from_img(img))

    return feature_vectors


def apply_lbph_to_whole_img(*args):
    from featureExtraction.lbph_fv_extraction import LBPHFeatureVectorExtraction
    from skimage.color import rgb2gray
    lbph = LBPHFeatureVectorExtraction()
    feature_vectors = []

    for img in args:
        temp_img = rgb2gray(img)
        feature_vectors.append(lbph.get_img_descriptor_from_img(temp_img))

    return feature_vectors


def apply_lbph_to_patched_img(img_pair, grid_size):
    import skimage
    from skimage.color import rgb2gray
    from skimage.util import view_as_blocks
    from featureVectorMerging.differential_comparison import DifferentialComparison

    temp_result = []

    img_pair[0] = rgb2gray(img_pair[0])
    img_pair[1] = rgb2gray(img_pair[1])

    img_size = 256
    margin = 256 % grid_size
    if margin != 0:
        if margin % 2 == 0:
            edge_crop = int(margin / 2)
            for i, img in enumerate(img_pair):
                img_pair[i] = skimage.util.crop(img, edge_crop)
        else:
            print("Image size and grid size are REALLY not compatible.")
            exit(-1)

    patch_shape = (int((img_size - margin) / grid_size), int((img_size - margin) / grid_size))
    patched_pair = [view_as_blocks(img_pair[0], patch_shape),
                    view_as_blocks(img_pair[1], patch_shape)]

    for i, row in enumerate(patched_pair[0]):
        for j, col in enumerate(patched_pair[0][i]):
            patch_pair = [patched_pair[0][i, j], patched_pair[1][i, j]]
            feature_vectors = apply_lbph_to_patch(patch_pair[0], patch_pair[1])

            euclidean = DifferentialComparison.calculate_euclidean_distance_from_binary_vector(feature_vectors)
            temp_result.append(euclidean)

    return temp_result


def apply_lbph_to_patch(*args):
    from featureExtraction.lbph_fv_extraction import LBPHFeatureVectorExtraction
    from skimage.color import rgb2gray
    lbph = LBPHFeatureVectorExtraction()
    feature_vectors = []

    for img in args:
        temp_img = rgb2gray(img)
        feature_vectors.append(lbph.get_img_descriptor_from_patch(temp_img))

    return feature_vectors
