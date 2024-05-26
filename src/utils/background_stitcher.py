import cv2
import numpy as np

from src.utils.video import get_next_frame


VIDEO_PATH = "../../resources/video/Walking.54138969.mp4"
OUTPUT_PATH = "../../output/background_stitched_recording.jpg"

COLUMNS_DIVISORS = 3
ROWS_DIVISORS = 3

UP_KEY = 119
DOWN_KEY = 115
RIGHT_KEY = 100
LEFT_KEY = 97
OK_KEY = 32
SAVE_KEY = 101
ESCAPE_KEY = 27

selected_patches = []


def calculate_region_from_pos(col: int, row: int, col_size: int, row_size: int, w: int, h: int) -> (int, int, int, int):
    min_x = max(0, col_size * col)
    min_y = max(0, row_size * row)

    max_x = min(min_x + col_size, w)
    max_y = min(min_y + row_size, h)

    return min_x, min_y, max_x, max_y


def draw_rects(frame: np.ndarray, current_col: int, current_row: int):
    h, w = frame.shape[:2]

    col_size = h // COLUMNS_DIVISORS
    row_size = w // ROWS_DIVISORS

    # Draw already selected region
    for col, row in selected_patches:
        min_x, min_y, max_x, max_y = calculate_region_from_pos(col, row, col_size, row_size, w, h)
        cv2.rectangle(frame, (min_x + 2, min_y + 2), (max_x - 2, max_y - 2), (255, 0, 255), 2)

    # Draw all regions
    for col in range(COLUMNS_DIVISORS):
        for row in range(ROWS_DIVISORS):
            min_x, min_y, max_x, max_y = calculate_region_from_pos(col, row, col_size, row_size, w, h)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)

    # Draw current region
    min_x, min_y, max_x, max_y = calculate_region_from_pos(current_col, current_row, col_size, row_size, w, h)
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)


def main():
    current_row = 0
    current_col = 0
    background_frame: np.ndarray | None = None

    for frame, fps in get_next_frame(VIDEO_PATH):
        keep_frame = True

        while keep_frame:
            new_frame = frame.copy()
            draw_rects(new_frame, current_col, current_row)

            cv2.imshow("Image", new_frame)
            key = cv2.waitKey(0)

            if key == ESCAPE_KEY:
                cv2.imwrite(OUTPUT_PATH, background_frame)
                return

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
                if (current_col, current_row) not in selected_patches:
                    selected_patches.append((current_col, current_row))

                if background_frame is None:
                    background_frame = np.zeros_like(frame)

                h, w = frame.shape[:2]

                col_size = h // COLUMNS_DIVISORS
                row_size = w // ROWS_DIVISORS

                min_x, min_y, max_x, max_y = calculate_region_from_pos(current_col, current_row, col_size, row_size, w,
                                                                       h)
                background_frame[min_y: max_y, min_x: max_x] = frame[min_y: max_y, min_x: max_x]


if __name__ == "__main__":
    main()
