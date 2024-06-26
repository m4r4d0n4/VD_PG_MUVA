import cv2
import numpy as np


def get_center_from_foreground(foreground_image: np.ndarray):
    """
    Returns the center point of the maximum contour of a binarized image.
    """
    foreground_image = cv2.Canny(foreground_image, 0, 100)

    foreground_image = cv2.dilate(foreground_image, np.ones((5, 5), np.uint8),
                                  iterations=2)

    contours, _ = cv2.findContours(foreground_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None

    max_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(max_contour)

    return x+w//2, y+h//2


def get_bounding_box_from_foreground(foreground_image: np.ndarray):
    """
    Returns the bounding box of the maximum contour of a binarized image.
    """
    foreground_image = cv2.Canny(foreground_image, 0, 100)

    foreground_image = cv2.dilate(foreground_image, np.ones((5, 5), np.uint8),
                                  iterations=2)

    contours, _ = cv2.findContours(foreground_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None, None, None

    max_contour = max(contours, key=cv2.contourArea)

    return cv2.boundingRect(max_contour)
