import cv2
import numpy as np

from src.background_subtraction.mog import MogBackgroundSubtraction
from src.utils.background_subtraction import clean_image, remove_shadows
from src.utils.video import get_next_frame
from src.techniques.lucas_kanade import LucasKanadeTracker


VIDEO_PATH = "../resources/video/Walking.60457274.mp4"


def run_lk_on_frame(lk: LucasKanadeTracker, frame: np.ndarray, bg_subtraction, init: bool, show_bbox: bool, show_flow: bool):
    foreground = bg_subtraction.apply(frame)
    foreground = remove_shadows(foreground)
    foreground = clean_image(foreground)

    f = frame.copy()

    if not init and cv2.countNonZero(foreground) > 10000:
        lk.initialize(frame, foreground)
        init = True

        # cv2.imshow("Lucas Kanade", foreground)
        # cv2.waitKey(0)
    if init:
        # Para ver la caja
        if show_bbox:
            f = lk.apply_bbox(frame)

        # Para el OF
        if show_flow:
            f = lk.apply(frame)

    return f, init


def main():
    # Initialize ParticleFilter
    lk = LucasKanadeTracker()
    
    # Initialize background subtractor
    bg_subtraction = MogBackgroundSubtraction()
    init = False
    for i, (frame, fps) in enumerate(get_next_frame(VIDEO_PATH)):
        f, init = run_lk_on_frame(lk, frame, bg_subtraction, init, True, False)

        cv2.imshow("Lucas Kanade", f)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
