import cv2

from src.background_subtraction.mog import MogBackgroundSubtraction
from src.utils.background_subtraction import clean_image, remove_shadows
from src.utils.video import get_next_frame


REMOVE_SHADOWS = True
REMOVE_NOISE = True
VIDEO_PATH = "../resources/video/Walking.54138969.mp4"

method = MogBackgroundSubtraction()

for frame, fps in get_next_frame(VIDEO_PATH):
    foreground = method.apply(frame)

    if REMOVE_SHADOWS:
        foreground = remove_shadows(foreground)

    if REMOVE_NOISE:
        foreground = clean_image(foreground, 2, 1)

    cv2.imshow("asd", foreground)
    cv2.waitKey(int(fps))
