import numpy as np


class DifferentialComparison:

    @staticmethod
    def get_differential_fv(reference_fv, bonafide_fv):
        return reference_fv - bonafide_fv

    @staticmethod
    def calculate_euclidean_distance(fv_a, fv_b):
        return np.linalg.norm(fv_a - fv_b)

    @staticmethod
    def is_pair_from_same_person(euclidean_distance):
        return euclidean_distance < 0.6


if __name__ == '__main__':
    reference = np.random.randint(10, size=128)
    bonafide = np.random.randint(10, size=128)

    differential = DifferentialComparison.get_differential_fv(reference, bonafide)

    print("Is differential comparison calculation correct?")

    if differential[0] == reference[0] - bonafide[0]:
        print(True)
    else:
        print(False)
