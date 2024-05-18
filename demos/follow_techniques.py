import cv2
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.utils.video import get_next_frame
from src.techniques.lucas_kanade import LucasKanadeTracker

LUCAS_KANADE = True

if LUCAS_KANADE:
    tracker = LucasKanadeTracker()
VIDEO_PATH = "../resources/video/Walking.60457274.mp4"
frame_init = True

for frame, fps in get_next_frame(VIDEO_PATH):
    if LUCAS_KANADE:
        if frame_init:
            tracker.initialize(frame)
            frame_init = False
        else:
            processed_frame = tracker.apply(frame)
            cv2.imshow('frame', processed_frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        
    