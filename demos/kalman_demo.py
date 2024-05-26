import cv2

from src.background_subtraction.mog import MogBackgroundSubtraction
from src.techniques.kalman import Kalman
from src.techniques.kalman_wrapper import KalmanWrapper
from src.utils.background_subtraction import clean_image, remove_shadows, fill_contours
from src.utils.get_center_from_foreground import get_bounding_box_from_foreground
from src.utils.video import get_next_frame


VIDEO_PATH = "../resources/video/Walking.54138969.mp4"


def main():
    pos_kalman = Kalman(1, 0, 0.9, 1)
    size_kalman = Kalman(1, 0, 0.5, 0.9)

    # Initialize background subtractor
    bg_subtractor = MogBackgroundSubtraction()

    kalman_wrapper = KalmanWrapper(pos_kalman, size_kalman)

    for frame, fps in get_next_frame(VIDEO_PATH):
        fg = bg_subtractor.apply(frame)

        # Reduce noise
        fg = clean_image(remove_shadows(fg))

        # Fill empty contours to improve the result
        fg = fill_contours(fg)

        x, y, w, h = get_bounding_box_from_foreground(fg)

        # Skip invalid frames
        if x is None:
            continue

        min_x, min_y, max_x, max_y = kalman_wrapper.apply(fg)

        # Draw the bounding boxes
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 255, 0), 3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Kalman", frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
