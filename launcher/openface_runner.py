import csv
import re

import numpy

from featureVectorMerging.differential_comparison import DifferentialComparison


def _load_csv(filename, filetype):
    fv = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if filetype == "labels":
                fv.append(row[1])
            elif filetype == "reps":
                fv.append([float(i) for i in row])
    return fv


def get_morphed_genuine_diff_fv(n_images):
    pattern_extract_genuine_ids_from_morphed = re.compile(r'([0-9]{5})')
    pairs = []

    for el in sorted_morphed_labels_fvs[0:n_images]:
        path = el[0]
        fv = el[1]
        genuine_id = re.findall(pattern_extract_genuine_ids_from_morphed, path)[0]
        pattern_current_genuine_id = re.compile('(.+' + genuine_id + '.+)')
        genuine_fv = next(
            x[1] for x in sorted_genuine_labels_fvs if pattern_current_genuine_id.search(x[0]) is not None)

        pairs.append(DifferentialComparison.get_differential_fv(numpy.array(fv), numpy.array(genuine_fv)))
    return pairs


def get_genuine_genuine_diff_fv(n_images):
    pairs = []
    pattern_extract_number_groups = re.compile('([0-9]{5})')

    for i, el in enumerate(sorted_feret_labels_fvs):
        path = el[0]
        fv = el[1]
        current_genuine_id = re.findall(pattern_extract_number_groups, path)[0]
        pattern_current_genuine_id = re.compile('\/' + current_genuine_id + 'f')

        next_i = i + 1
        while pattern_current_genuine_id.search(sorted_feret_labels_fvs[next_i][0]) is not None:
            pairs.append(DifferentialComparison.get_differential_fv(numpy.array(fv),
                                                                    numpy.array(sorted_feret_labels_fvs[next_i][1])))
            next_i += 1
            if len(pairs) >= n_images:
                return pairs

    return pairs


def join_and_sort(a, b):
    result = []
    if len(a) == len(b):
        for i in range(len(a)):
            result.append([a[i], b[i]])

    return sorted(result, key=lambda x: x[0])


biometix_genuine_labels = _load_csv("../assets/data-of/biometix-genuine/labels.csv", "labels")
biometix_genuine_fv = _load_csv("../assets/data-of/biometix-genuine/reps.csv", "reps")
sorted_genuine_labels_fvs = join_and_sort(biometix_genuine_labels, biometix_genuine_fv)

biometix_morphed_labels = _load_csv("../assets/data-of/biometix-morphed/labels.csv", "labels")
biometix_morphed_fv = _load_csv("../assets/data-of/biometix-morphed/reps.csv", "reps")
sorted_morphed_labels_fvs = join_and_sort(biometix_morphed_labels, biometix_morphed_fv)

feret_labels = _load_csv("../assets/data-of/feret/labels.csv", "labels")
feret_fv = _load_csv("../assets/data-of/feret/reps.csv", "reps")
sorted_feret_labels_fvs = join_and_sort(feret_labels, feret_fv)

if __name__ == '__main__':
    print(len(feret_fv))
    print(len(feret_labels))
    print(numpy.column_stack((feret_labels, feret_fv)))
    print(sorted_feret_labels_fvs[0])
    # sorted_feret_labels_fvs2 = numpy.column_stack((feret_labels, feret_fv)).tolist()
    # print(numpy.array(sorted_feret_labels_fvs)[0:10][:, 0])
    # print(numpy.array(sorted_feret_labels_fvs2)[0:10][:, 0])
    # for el in sorted_feret_labels_fvs2:
    #     if el[0] == sorted_feret_labels_fvs[0][0]:
    #         print(el[1] == sorted_feret_labels_fvs[0][1])
