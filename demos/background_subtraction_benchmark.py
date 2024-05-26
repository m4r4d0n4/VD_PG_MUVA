import time

import cv2

from src.background_subtraction.knn import KNNBackgroundSubtraction
from src.background_subtraction.mog import MogBackgroundSubtraction
from src.background_subtraction.static_background_subtraction import StaticBackgroundSubtraction
from src.utils.video import get_next_frame

VIDEO_PATH = "../resources/video/Walking.54138969.mp4"
STITCHED_BACKGROUND_PATH = "../output/background_stitched.jpg"

# Load stitched background
stitched_background = cv2.imread(STITCHED_BACKGROUND_PATH)

# Prepare models (missing first_frame model)
stitch_method = StaticBackgroundSubtraction(stitched_background)
mog_method = MogBackgroundSubtraction()
knn_method = KNNBackgroundSubtraction()

# Define methods to loop over
methods = ["first_frame", "stitch", "mog", "knn"]
method_instances = [None, stitch_method, mog_method, knn_method]

for method_name in methods:
    frames = get_next_frame(VIDEO_PATH)
    first_frame, _ = frames.__next__()  # Always skip the first frame

    method = method_instances.pop(0)

    # Lazy initialize the first-frame method manually
    if method_name == "first_frame":
        method = StaticBackgroundSubtraction(first_frame)

    # Start benchmark
    time_start = time.perf_counter()

    for frame, _ in frames:
        _ = method.apply(frame)

    time_end = time.perf_counter()
    process_time = time_end - time_start

    print(f"Method '{method_name}' took {process_time:.2f} seconds.")
