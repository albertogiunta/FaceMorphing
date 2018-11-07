import re

import numpy

from featureVectorMerging.differential_comparison import DifferentialComparison
from launcher import sorted_biometix_morphed_labels_fvs, sorted_biometix_genuine_labels_fvs, \
    sorted_feret_genuine_labels_fvs


def get_morphed_genuine_diff_fv(n_images):
    pattern_extract_genuine_ids_from_morphed = re.compile(r'([0-9]{5})')
    pairs = []

    for el in sorted_biometix_morphed_labels_fvs[0:n_images]:
        path = el[0]
        fv = el[1]
        genuine_id = re.findall(pattern_extract_genuine_ids_from_morphed, path)[0]
        pattern_current_genuine_id = re.compile('(.+' + genuine_id + '.+)')
        genuine_fv = next(
            x[1] for x in sorted_biometix_genuine_labels_fvs if pattern_current_genuine_id.search(x[0]) is not None)

        pairs.append(DifferentialComparison.get_differential_fv(numpy.array(fv), numpy.array(genuine_fv)))
    return pairs


def get_genuine_genuine_diff_fv(n_images):
    pairs = []
    pattern_extract_number_groups = re.compile('([0-9]{5})')

    for i, el in enumerate(sorted_feret_genuine_labels_fvs):
        path = el[0]
        fv = el[1]
        current_genuine_id = re.findall(pattern_extract_number_groups, path)[0]
        pattern_current_genuine_id = re.compile('\/' + current_genuine_id + 'f')

        next_i = i + 1
        while pattern_current_genuine_id.search(sorted_feret_genuine_labels_fvs[next_i][0]) is not None:
            pairs.append(DifferentialComparison.get_differential_fv(numpy.array(fv),
                                                                    numpy.array(
                                                                        sorted_feret_genuine_labels_fvs[next_i][1])))
            next_i += 1
            if len(pairs) >= n_images:
                return pairs

    return pairs




if __name__ == '__main__':
    print("ciao")
    # print(len(feret_fv))
    # print(len(feret_labels))
    # print(numpy.column_stack((feret_labels, feret_fv)))
    # print(sorted_feret_labels_fvs[0])

    # sorted_feret_labels_fvs2 = numpy.column_stack((feret_labels, feret_fv)).tolist()
    # print(numpy.array(sorted_feret_labels_fvs)[0:10][:, 0])
    # print(numpy.array(sorted_feret_labels_fvs2)[0:10][:, 0])
    # for el in sorted_feret_labels_fvs2:
    #     if el[0] == sorted_feret_labels_fvs[0][0]:
    #         print(el[1] == sorted_feret_labels_fvs[0][1])
