import numpy as np


class DifferentialComparison:

    @staticmethod
    def get_differential_fv(reference_fv, bonafide_fv):
        return reference_fv - bonafide_fv


if __name__ == '__main__':
    reference = np.random.randint(10, size=128)
    bonafide = np.random.randint(10, size=128)

    differential = DifferentialComparison.get_differential_fv(reference, bonafide)

    print("Is differential comparison calculation correct?")

    if differential[0] == reference[0] - bonafide[0]:
        print(True)
    else:
        print(False)
