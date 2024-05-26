import cv2

from src.background_subtraction.mog import MogBackgroundSubtraction
from src.background_subtraction.knn import KNNBackgroundSubtraction
from src.utils.background_subtraction import clean_image, remove_shadows
from src.utils.video import get_next_frame


REMOVE_SHADOWS = True
REMOVE_NOISE = True

BG_MOG = False
BG_KNN = True

VIDEO_PATH = "../resources/video/Walking.54138969.mp4"

if BG_MOG:
    method = MogBackgroundSubtraction()
elif BG_KNN:
    method = KNNBackgroundSubtraction()

for frame, fps in get_next_frame(VIDEO_PATH):
    foreground = method.apply(frame)

    if REMOVE_SHADOWS:
        foreground = remove_shadows(foreground)

    if REMOVE_NOISE:
        foreground = clean_image(foreground)

    # Display frame
    cv2.imshow("MOG Background Subtraction", foreground)
    cv2.waitKey(int(fps))
