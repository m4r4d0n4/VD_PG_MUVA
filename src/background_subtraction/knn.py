import cv2


class KNNBackgroundSubtraction:
    """
    Class to handle KNN background subtraction on images.
    """
    def __init__(self):
        self.knn = cv2.createBackgroundSubtractorKNN()

    def apply(self, frame):
        return self.knn.apply(frame)
