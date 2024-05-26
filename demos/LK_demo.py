import cv2
import sys,os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.background_subtraction.mog import MogBackgroundSubtraction
from src.techniques.particle_filter_v2 import ParticleFilterV2, disperse_motion_model
from src.techniques.particle_filter_wrapper import ParticleFilterWrapper
from src.utils.background_subtraction import clean_image, remove_shadows
from src.utils.video import get_next_frame
from src.techniques.lucas_kanade import LucasKanadeTracker

VIDEO_PATH = "./resources/video/Walking.60457274.mp4"


def main():
    # Initialize ParticleFilter
    lk = LucasKanadeTracker()
    
    # Initialize background subtractor
    bg_subtraction = MogBackgroundSubtraction()
    init = False
    for frame, fps in get_next_frame(VIDEO_PATH):
        foreground = bg_subtraction.apply(frame)
        foreground = remove_shadows(foreground)
        foreground = clean_image(foreground)
        #print(cv2.countNonZero(foreground))
        if not init and cv2.countNonZero(foreground) > 10000:
            lk.initialize(frame,foreground)
            cv2.imshow("asd", foreground)
            cv2.waitKey(0)
            init = True
        if init:
            #Para ver la caja
            f = lk.apply_bbox(frame)
            #Para el OF
            #f = lk.apply(frame)
            cv2.imshow("Lucas-Kanade", f)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
