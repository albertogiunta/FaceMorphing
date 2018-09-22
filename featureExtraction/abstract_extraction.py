import glob
from abc import abstractmethod

import numpy as np


class AbstractFVExtraction:

    def get_img_pair_descriptors(self, faces_pair_folder_path):
        descriptors = []

        images = glob.glob(faces_pair_folder_path)

        if len(images) > 2:
            print("ERROR: There are more than 2 pictures in the specified folder ({})".format(faces_pair_folder_path))
            exit(0)

        for img_path in images:
            descriptors.append(self.get_img_descriptor(img_path))

        return np.array(descriptors)

    @abstractmethod
    def get_img_descriptor(self, img_path):
        pass
