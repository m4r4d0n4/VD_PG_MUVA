import cv2
import numpy as np


def remove_shadows(foreground: np.ndarray, shadow_value: int = 150) -> np.ndarray:
    ret, binary_mask = cv2.threshold(foreground, shadow_value, 255, cv2.THRESH_BINARY)

    return cv2.bitwise_and(foreground, binary_mask)


def clean_image(foreground: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    eroded_mask = cv2.erode(foreground, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
    dilated_mask = cv2.dilate(eroded_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)

    return cv2.bitwise_and(foreground, dilated_mask)
