import cv2
import numpy as np


class StaticBackgroundSubtraction:
    """
    Class to handle static background subtraction on images.
    """
    def __init__(self, background: np.ndarray, threshold: float = 0.1):
        self.background = self._normalize_frame(background)
        self.threshold = threshold

    def apply(self, frame):
        diff = np.abs(self._normalize_frame(frame) - self.background)

        _, thres = cv2.threshold(diff, self.threshold, 1, cv2.THRESH_BINARY)

        return thres * 255

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return new_frame.astype(np.float32) / 255
