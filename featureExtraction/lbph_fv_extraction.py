import matplotlib.pyplot as plt
import numpy as np
from skimage import feature as sk
from skimage.util import view_as_blocks

import utils.config_utils as config_utils
import utils.img_utils as img_utils
from featureExtraction.abstract_extraction import AbstractFVExtraction


class LBPHFeatureVectorExtraction(AbstractFVExtraction):

    def __init__(self):
        self.conf = config_utils.get_config("FVExtractionLbph")
        self.img = None
        self.neighbours = self.conf["neighbours"]
        self.radius = self.conf["radius"]
        self.grid_size = self.conf["gridSize"]
        self.img_size = self.conf["imgSize"]
        self.n_rows = self.conf["nRows"]
        self.n_cols = self.conf["nCols"]
        self.nbp_method_name = self.conf["lbpMethodName"]
        self.nbp_method_colors = self.conf["lbpMethodColors"]

    def get_img_descriptor_from_img(self, img):
        self.img = img
        return self._get_img_descriptor(False)

    def get_img_descriptor_from_patch(self, patch):
        self.img = patch
        return self._get_img_descriptor(True)

    def get_img_descriptor_from_path(self, img_path):
        self.img = img_utils.load_img_skimage(img_path)
        return self._get_img_descriptor()

    def _get_img_descriptor(self, use_img_as_patch):
        # img_utils.show_img_skimage(self.img)

        feature_vector = np.array([])
        lbp_img = sk.local_binary_pattern(self.img, self.neighbours, self.radius, method=self.nbp_method_name)

        if use_img_as_patch:
            feature_vector = np.append(feature_vector, self._get_histogram(lbp_img))
        else:
            lbp_img_as_blocks = \
                view_as_blocks(lbp_img, (int(self.img_size / self.grid_size), int(self.img_size / self.grid_size)))
            for row in range(self.n_rows):
                for col in range(self.n_cols):
                    curr_block = lbp_img_as_blocks[row, col]
                    feature_vector = np.append(feature_vector, self._get_histogram(curr_block))

        return feature_vector

    def _get_histogram(self, block):
        # hist will be 255 elements long because 255 is the number of colors used as bins
        hist, _ = np.histogram(block, bins=np.arange(0, self.nbp_method_colors))
        # self._show_block_histogram(block)
        return hist

    def _show_block_histogram(self, block):
        plt.hist(block.ravel(), bins=np.arange(0, self.nbp_method_colors))
        plt.show()


if __name__ == '__main__':
    lbph = LBPHFeatureVectorExtraction()

    feature_vector = lbph.get_img_descriptor_from_path(
        img_utils.compile_img_path(img_name="00002_930831_fa.png", img_folder="biometix/genuine"))
    print(feature_vector)

    feature_vector2 = lbph.get_img_descriptor_from_path(
        img_utils.compile_img_path(img_name="M_00002_00320.jpg", img_folder="biometix/morphed"))
    print(feature_vector2)
