import numpy as np

from src.techniques.kalman import Kalman
from src.utils.get_center_from_foreground import get_bounding_box_from_foreground


class KalmanWrapper:
    def __init__(self, pos_kalman: Kalman, size_kalman: Kalman):
        self.pos_kalman = pos_kalman
        self.size_kalman = size_kalman

    def apply(self, frame: np.ndarray) -> (int, int, int, int):
        # Get measurements
        x, y, w, h = get_bounding_box_from_foreground(frame)

        if x is None:
            return None, None, None, None

        # Calculate predictions
        pred_x, pred_y = self.pos_kalman.predict()
        pred_x, pred_y = int(pred_x), int(pred_y)

        pred_w, pred_h = self.size_kalman.predict()
        pred_w, pred_h = int(pred_w), int(pred_h)

        # Update the models with the observed state
        self.pos_kalman.update((x, y))
        self.size_kalman.update((w, h))

        return pred_x, pred_y, pred_x + pred_w, pred_y + pred_h

        """
        # Draw the bounding boxes
        cv2.rectangle(frame, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (255, 255, 0), 3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Kalman", frame)
        cv2.waitKey(0)
        """