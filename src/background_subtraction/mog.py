import cv2


class MogBackgroundSubtraction:
    """
    Class to handle MOG background subtraction on images.
    """
    def __init__(self):
        self.mog = cv2.createBackgroundSubtractorMOG2()

    def apply(self, frame):
        return self.mog.apply(frame)
