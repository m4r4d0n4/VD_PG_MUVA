import cv2
import numpy as np

from src.background_subtraction.mog import MogBackgroundSubtraction
from src.techniques.kalman import Kalman
from src.utils.background_subtraction import clean_image, remove_shadows, fill_contours
from src.utils.get_center_from_foreground import get_center_from_foreground, get_bounding_box_from_foreground
from src.utils.video import get_next_frame

pos_kalman = Kalman(1, 0, 0.9, 1)
size_kalman = Kalman(1, 0, 0.5, 0.9)

# Initialize background subtractor
bg_subtraction = MogBackgroundSubtraction()

measurements = []
predicted = []

VIDEO_PATH = "../resources/video/Walking.54138969.mp4"

for frame, fps in get_next_frame(VIDEO_PATH):
    fg = bg_subtraction.apply(frame)

    # Reduce noise
    fg = clean_image(remove_shadows(fg))

    # Fill empty contours to improve the result
    fg = fill_contours(fg)

    # Get measurements
    x, y, w, h = get_bounding_box_from_foreground(fg)

    if x is None:
        continue

    # Calculate predictions
    pred_x, pred_y = pos_kalman.predict()
    pred_x, pred_y = int(pred_x), int(pred_y)

    pred_w, pred_h = size_kalman.predict()
    pred_w, pred_h = int(pred_w), int(pred_h)

    # Update the models with the observed state
    pos_kalman.update((x, y))
    size_kalman.update((w, h))

    # Draw the bounding boxes
    cv2.rectangle(frame, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (255, 255, 0), 3)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Kalman", frame)
    cv2.waitKey(0)
