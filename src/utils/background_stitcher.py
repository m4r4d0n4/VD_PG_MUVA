import cv2
import numpy as np

from src.utils.video import get_next_frame


VIDEO_PATH = "../../resources/video/Walking.54138969.mp4"
OUTPUT_PATH = "../../output/background_stitched.jpg"

COLUMNS_DIVISORS = 3
ROWS_DIVISORS = 3

current_row = 0
current_col = 0

UP_KEY = 119
DOWN_KEY = 115
RIGHT_KEY = 100
LEFT_KEY = 97
OK_KEY = 32
SAVE_KEY = 101
ESCAPE_KEY = 27

background_frame: np.ndarray = None

def draw_rects(frame):
    h, w = frame.shape[:2]

    col_size = h // COLUMNS_DIVISORS
    row_size = w // ROWS_DIVISORS

    for col in range(COLUMNS_DIVISORS):
        for row in range(ROWS_DIVISORS):
            min_x = max(0, col_size * col)
            min_y = max(0, row_size * row)

            max_x = min(min_x + col_size, w)
            max_y = min(min_y + row_size, h)

            color = (255, 0, 0)
            if col == current_col and row == current_row:
                color = (0, 255, 0)

            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 2)

    min_x = max(0, col_size * current_col)
    min_y = max(0, row_size * current_row)

    max_x = min(min_x + col_size, w)
    max_y = min(min_y + row_size, h)

    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)


for frame, fps in get_next_frame(VIDEO_PATH):
    keep_frame = True

    while keep_frame:
        new_frame = frame.copy()
        draw_rects(new_frame)

        cv2.imshow("Image", new_frame)
        key = cv2.waitKey(0)

        if key == ESCAPE_KEY:
            cv2.imwrite(OUTPUT_PATH, background_frame)
            exit(0)  # FORCE EXIT

        if key == UP_KEY:
            current_row = max(0, current_row - 1)

        if key == DOWN_KEY:
            current_row = min(ROWS_DIVISORS - 1, current_row + 1)

        if key == LEFT_KEY:
            current_col = max(0, current_col - 1)

        if key == RIGHT_KEY:
            current_col = min(COLUMNS_DIVISORS - 1, current_col + 1)

        if key == OK_KEY:
            keep_frame = False

        if key == SAVE_KEY:
            if background_frame is None:
                background_frame = np.zeros_like(frame)

            h, w = frame.shape[:2]

            col_size = h // COLUMNS_DIVISORS
            row_size = w // ROWS_DIVISORS

            min_x = max(0, col_size * current_col)
            min_y = max(0, row_size * current_row)

            max_x = min(min_x + col_size, w)
            max_y = min(min_y + row_size, h)

            background_frame[min_y: max_y, min_x: max_x] = frame[min_y: max_y, min_x: max_x]
