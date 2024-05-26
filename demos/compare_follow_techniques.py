import cv2

from src.background_subtraction.mog import MogBackgroundSubtraction
from src.techniques.kalman import Kalman
from src.techniques.kalman_wrapper import KalmanWrapper
from src.techniques.particle_filter_v2 import ParticleFilterV2, disperse_motion_model
from src.techniques.particle_filter_wrapper import ParticleFilterWrapper
from src.utils.background_subtraction import clean_image, remove_shadows, fill_contours
from src.utils.get_center_from_foreground import get_bounding_box_from_foreground
from src.utils.video import get_next_frame


VIDEO_PATH = "../resources/video/Walking.54138969.mp4"

# Initialize background subtractor
bg_subtractor = MogBackgroundSubtraction()

# Initialize Kalman
pos_kalman = Kalman(1, 0, 0.9, 1)
size_kalman = Kalman(1, 0, 0.5, 0.9)
kalman_wrapper = KalmanWrapper(pos_kalman, size_kalman)

# Initialize Particle Filter
num_particles = 1000
pf = ParticleFilterV2(num_particles, disperse_motion_model)
pfw = ParticleFilterWrapper(bg_subtractor, pf)

for frame, fps in get_next_frame(VIDEO_PATH):
    # Apply the background subtractor
    fg = bg_subtractor.apply(frame)

    # Reduce noise
    fg = clean_image(remove_shadows(fg))

    # Fill empty contours to improve the result
    fg = fill_contours(fg)

    x, y, w, h = get_bounding_box_from_foreground(fg)

    # Skip invalid frames
    if x is None:
        continue

    # Draw the observed bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Run Kalman
    k_min_x, k_min_y, k_max_x, k_max_y = kalman_wrapper.apply(fg)
    cv2.rectangle(frame, (k_min_x, k_min_y), (k_max_x, k_max_y), (255, 255, 0), 2)

    # Run Particle Filter
    pf_min_x, pf_min_y, pf_max_x, pf_max_y = pfw.apply(fg)
    cv2.rectangle(frame, (pf_min_x, pf_min_y), (pf_max_x, pf_max_y), (255, 0, 255), 2)

    # Run Lucas-Kanade
    # TODO

    # Run YOLO
    # TODO

    cv2.imshow("Kalman", frame)
    cv2.waitKey(0)
