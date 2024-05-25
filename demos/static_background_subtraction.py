import cv2

from src.background_subtraction.static_background_subtraction import StaticBackgroundSubtraction
from src.utils.video import get_next_frame

VIDEO_PATH = "../resources/video/Walking.54138969.mp4"
USE_FIRST_FRAME = False
STITCHED_BACKGROUND_PATH = "../output/background_stitched.jpg"


frames = get_next_frame(VIDEO_PATH)

if USE_FIRST_FRAME:
    base_background, _ = frames.__next__()
else:
    base_background = cv2.imread(STITCHED_BACKGROUND_PATH)

method = StaticBackgroundSubtraction(base_background)

for frame, fps in frames:
    fg = method.apply(frame)

    cv2.imshow("Static Background Subtraction", fg)
    cv2.waitKey(int(fps))
