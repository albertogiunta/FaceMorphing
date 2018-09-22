import matplotlib.pyplot as plt
import numpy as np
from skimage import feature as sk
from skimage.util import view_as_blocks

import utils.img_utils as img_utils
from featureExtraction.abstract_extraction import AbstractFVExtraction


class LBPHFeatureVectorExtraction(AbstractFVExtraction):

    def __init__(self):
        self.neighbours = 8
        self.radius = 1
        self.grid_size = (8, 8)
        self.img_size = (256, 256)
        self.n_rows = int(self.img_size[0] / self.grid_size[0])
        self.n_cols = int(self.img_size[1] / self.grid_size[1])
        self.nbp_method = ("default", 256)  # number of possible "colors", default = (0 to 255)

    def get_img_descriptor(self, img_path):
        img = self.load_img_skimage(img_path)
        img_utils.show_img_skimage(img)

        lbp_img = sk.local_binary_pattern(img, self.neighbours, self.radius, method=self.nbp_method[0])
        lbp_img_as_blocks = view_as_blocks(lbp_img, self.grid_size)

        feature_vector = np.array([])

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                curr_block = lbp_img_as_blocks[row, col]
                hist, _ = np.histogram(curr_block.ravel(), bins=np.arange(0, self.nbp_method[1]))
                feature_vector = np.append(feature_vector, hist)
                # self._show_block_histogram(curr_block)

        return feature_vector

    def _show_block_histogram(self, block):
        plt.hist(block.ravel(), bins=np.arange(0, self.nbp_method[1]))
        plt.show()


if __name__ == '__main__':
    lbph = LBPHFeatureVectorExtraction()
    feature_vector = lbph.get_img_descriptor(img_utils.compile_img_path(img_name="george1.png", img_folder="img"))
    print(feature_vector)
