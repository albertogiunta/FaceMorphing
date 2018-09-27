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
        return self._get_img_descriptor()

    def get_img_descriptor_from_path(self, img_path):
        self.img = img_utils.load_img_skimage(img_path)
        return self._get_img_descriptor()

    def _get_img_descriptor(self):
        # img_utils.show_img_skimage(self.img)

        lbp_img = sk.local_binary_pattern(self.img, self.neighbours, self.radius, method=self.nbp_method_name)
        lbp_img_as_blocks = view_as_blocks(lbp_img, (self.grid_size, self.grid_size))

        feature_vector = np.array([])

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                curr_block = lbp_img_as_blocks[row, col]
                hist, _ = np.histogram(curr_block.ravel(), bins=np.arange(0, self.nbp_method_colors))
                feature_vector = np.append(feature_vector, hist)
                # self._show_block_histogram(curr_block)

        return feature_vector

    def _show_block_histogram(self, block):
        plt.hist(block.ravel(), bins=np.arange(0, self.nbp_method_colors))
        plt.show()


if __name__ == '__main__':
    lbph = LBPHFeatureVectorExtraction()

    feature_vector = lbph.get_img_descriptor_from_path(
        img_utils.compile_img_path(img_name="00002_930831_fa.png", img_folder="biometrix/genuine"))
    print(feature_vector)

    feature_vector2 = lbph.get_img_descriptor_from_path(
        img_utils.compile_img_path(img_name="M_00002_00320.jpg", img_folder="biometrix/morphed"))
    print(feature_vector2)
