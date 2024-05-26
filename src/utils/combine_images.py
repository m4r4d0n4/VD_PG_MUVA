import cv2
import numpy as np


def combine_images_with_borders(images: list[np.ndarray], border_width: int = 5) -> np.ndarray:
    """
    Stacks multiple images horizontally with a white border between them.
    """
    shape = [*images[0].shape]
    shape[1] = border_width

    border = np.ones(shape) * 255

    all_images = []
    for image in images:
        all_images.append(image)
        all_images.append(border)

    all_images.pop(-1)

    return np.hstack(all_images)


def resize_to_height(image: np.ndarray, new_height: int) -> np.ndarray:
    """
    Returns the image resized to a new height while keeping the image ratio intact.
    """
    h, w = image.shape[:2]

    aspect_ratio = w / h
    new_width = int(new_height * aspect_ratio)

    return cv2.resize(image, (new_width, new_height))
