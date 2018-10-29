def compute_accuracy(probabilities, labels):
    accuracy = 0
    for i in range(len(labels)):
        if labels[i] == probabilities[i].argmax():
            accuracy += 1
    return accuracy / len(labels)


def compute_accuracy_from_distances(distances, labels):
    accuracy = 0
    for i in range(len(labels)):
        if labels[i] == distances[i].argmin():
            accuracy += 1
    return accuracy / len(labels)


def compute_acceptance_rate(scores, thr):
    return sum(1 for s in scores if s >= thr) / len(scores)


def compute_acceptance_rate_from_distances(distances, thr):
    return sum(1 for d in distances if d <= thr) / len(distances)


def compute_rejection_rate(scores, thr):
    return sum(1 for s in scores if s < thr) / len(scores)


def compute_rejection_rate_from_distances(distances, thr):
    return sum(1 for d in distances if d > thr) / len(distances)


def compute_frr_at_given_far(genuine_scores, impostor_scores, far):
    sorted_different_thresholds = list(set(genuine_scores + impostor_scores))
    sorted_different_thresholds.sort(reverse=True)

    far_thr = None
    for i in range(len(sorted_different_thresholds)):
        thr = sorted_different_thresholds[i]
        current_far = compute_acceptance_rate(impostor_scores, thr)
        if current_far > far:
            if i > 0:
                far_thr = sorted_different_thresholds[i - 1]
            break

    if far_thr is None:
        if i == len(sorted_different_thresholds) - 1:
            far_thr = sorted_different_thresholds[i]
        else:
            return None, None

    return compute_rejection_rate(genuine_scores, far_thr), compute_acceptance_rate(impostor_scores, far_thr)


def compute_frr_at_given_far_from_distances(genuine_distances, impostor_distances, far):
    sorted_different_thresholds = list(set(genuine_distances + impostor_distances))
    sorted_different_thresholds.sort(reverse=False)

    far_thr = None
    for i in range(len(sorted_different_thresholds)):
        thr = sorted_different_thresholds[i]
        current_far = compute_acceptance_rate_from_distances(impostor_distances, thr)
        if current_far > far:
            if i > 0:
                far_thr = sorted_different_thresholds[i - 1]
            break

    if far_thr is None:
        if i == len(sorted_different_thresholds) - 1:
            far_thr = sorted_different_thresholds[i]
        else:
            return None, None

    return compute_rejection_rate_from_distances(genuine_distances, far_thr), compute_acceptance_rate_from_distances(
        impostor_distances, far_thr)


def get_genuine_impostor_scores(probabilities, labels, genuine_label=0):
    genuine_scores = list()
    impostor_scores = list()
    for i in range(len(labels)):
        if labels[i] == genuine_label:
            genuine_scores.append(probabilities[i][genuine_label])
        else:
            impostor_scores.append(probabilities[i][genuine_label])

    return genuine_scores, impostor_scores


def compute_frr_at_given_far_from_probabilities(probabilities, far, labels, genuine_label=0):
    genuine_scores, impostor_scores = get_genuine_impostor_scores(probabilities, labels, genuine_label)

    frr, far_v = compute_frr_at_given_far(genuine_scores, impostor_scores, far)

    return frr, far_v
