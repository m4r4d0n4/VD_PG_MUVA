import cv2

from demos.LK_demo import run_lk_on_frame
from src.background_subtraction.mog import MogBackgroundSubtraction
from src.techniques.kalman import Kalman
from src.techniques.kalman_wrapper import KalmanWrapper
from src.techniques.lucas_kanade import LucasKanadeTracker
from src.techniques.particle_filter_v2 import ParticleFilterV2, disperse_motion_model
from src.techniques.particle_filter_wrapper import ParticleFilterWrapper
from src.techniques.yolo_person_detector import YOLOPersonDetector
from src.utils.background_subtraction import clean_image, remove_shadows, fill_contours
from src.utils.combine_images import combine_images_with_borders, resize_to_height
from src.utils.get_center_from_foreground import get_bounding_box_from_foreground
from src.utils.video import get_next_frame, create_video_as_other


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
pfw = ParticleFilterWrapper(pf)

# Initialize Lucas-Kanade
lk = LucasKanadeTracker()
bg_subtraction = MogBackgroundSubtraction()
is_lk_ready = False

# Initialize YOLO
detector = YOLOPersonDetector()

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
    observed_frame = frame.copy()
    cv2.rectangle(observed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Run Kalman
    k_min_x, k_min_y, k_max_x, k_max_y = kalman_wrapper.apply(fg)
    kalman_frame = frame.copy()
    cv2.rectangle(kalman_frame, (k_min_x, k_min_y), (k_max_x, k_max_y), (255, 255, 0), 2)

    # Run Particle Filter
    pf_min_x, pf_min_y, pf_max_x, pf_max_y = pfw.apply(fg)
    pf_frame = frame.copy()
    cv2.rectangle(pf_frame, (pf_min_x, pf_min_y), (pf_max_x, pf_max_y), (255, 0, 255), 2)

    # Run Lucas-Kanade
    lk_frame, is_lk_ready = run_lk_on_frame(lk, frame.copy(), bg_subtraction, is_lk_ready, True, False)

    # Run YOLO
    _, _, _, yolo_x1, yolo_y1, yolo_x2, yolo_y2, _ = detector.apply(frame)
    yolo_frame = frame.copy()
    cv2.rectangle(yolo_frame, (int(yolo_x1), int(yolo_y1)), (int(yolo_x2), int(yolo_y2)), (0, 255, 255), 2)

    # Combine all the frames into one
    combined = combine_images_with_borders([
        observed_frame / 255,
        kalman_frame / 255,
        pf_frame / 255,
        lk_frame / 255,
        yolo_frame / 255
    ])
    combined = resize_to_height(combined, 350)

    cv2.imshow("Kalman", combined)
    cv2.waitKey(29)

cv2.destroyAllWindows()
