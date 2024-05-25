import cv2
import numpy as np


def remove_shadows(foreground: np.ndarray, shadow_value: int = 150) -> np.ndarray:
    """
    Removes the shadows from a foreground image generated using some OpenCV methods.
    :param foreground: Image
    :param shadow_value: The threshold grayscale value to consider shadow
    :return: The image without the shadow pixels
    """
    ret, binary_mask = cv2.threshold(foreground, shadow_value, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_and(foreground, binary_mask)


def clean_image(foreground: np.ndarray, kernel_size: int = 2, iterations: int = 1) -> np.ndarray:
    """
    Removes noise on the image by using a morphological opening.
    :param foreground: The image
    :param kernel_size: The size of the kernel to apply the morphological opening
    :param iterations: Number of iterations
    :return: The image without the noise
    """
    eroded_mask = cv2.erode(foreground, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
    dilated_mask = cv2.dilate(eroded_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)

    return cv2.bitwise_and(foreground, dilated_mask)


def fill_contours(foreground: np.ndarray):
    contours, _ = cv2.findContours(foreground, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(foreground, contours, -1, color=(255), thickness=cv2.FILLED)
