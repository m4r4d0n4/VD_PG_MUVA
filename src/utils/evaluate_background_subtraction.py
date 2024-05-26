

def evaluate_background_subtraction(gt, other):
    """
    Helper function to evaluate the background subtraction methods.
    Compares the gt image with another one.
    :returns: A dictionary with precision, recall and f1_score.
    """
    good_fg_detected = 0

    total_fg_detected = 0
    total_fg_gt = 0

    h, w = gt.shape[:2]
    for x in range(w):
        for y in range(h):
            gt_pixel = gt[y, x]
            pixel = other[y, x]

            if pixel == 255:
                total_fg_detected += 1

            if gt_pixel == 255:
                total_fg_gt += 1

                if pixel == 255:
                    good_fg_detected += 1

    if total_fg_detected == 0:
        precision = 0
    else:
        precision = good_fg_detected / total_fg_detected

    recall = good_fg_detected / total_fg_gt

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (recall * precision) / (recall + precision)

    return {
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score,
    }
