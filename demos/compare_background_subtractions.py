import cv2
import matplotlib.pyplot as plt

from src.background_subtraction.knn import KNNBackgroundSubtraction
from src.background_subtraction.mog import MogBackgroundSubtraction
from src.background_subtraction.static_background_subtraction import StaticBackgroundSubtraction
from src.utils.combine_images import combine_images_with_borders, resize_to_height
from src.utils.evaluate_background_subtraction import evaluate_background_subtraction
from src.utils.video import get_next_frame


"""
WARNING
This script is slow when running it with GENERATE_GRAPHS active,
if you just want to see the outputs in a video, set the GENERATE_GRAPHS variable to False
"""

VIDEO_PATH = "../resources/video/Walking.54138969.mp4"
GT_PATH = "../resources/bs/Walking.54138969.mp4"

GENERATE_GRAPHS = False
OUTPUT_VIDEO_COMPARISONS = True
STITCHED_BACKGROUND_PATH = "../output/background_stitched.jpg"


def main():
    frames = get_next_frame(VIDEO_PATH)
    gt_frames = get_next_frame(GT_PATH)

    first_frame, _ = frames.__next__()
    stitched_background = cv2.imread(STITCHED_BACKGROUND_PATH)
    gt_frames.__next__()  # Discard the first frame of the gt too

    first_frame_method = StaticBackgroundSubtraction(first_frame)
    stitch_method = StaticBackgroundSubtraction(stitched_background)
    mog_method = MogBackgroundSubtraction()
    knn_method = KNNBackgroundSubtraction()

    precisions = [[], [], [], []]
    recalls = [[], [], [], []]
    f1_scores = [[], [], [], []]
    frames_count = 0

    scores_step = 10

    for i, (frame, fps) in enumerate(frames):
        frames_count += 1

        gt_fg, _ = gt_frames.__next__()
        gt_fg_gray = cv2.cvtColor(gt_fg, cv2.COLOR_BGR2GRAY)

        ff_fg = first_frame_method.apply(frame)
        st_fg = stitch_method.apply(frame)
        mog_fg = mog_method.apply(frame)
        knn_fg = knn_method.apply(frame)

        if OUTPUT_VIDEO_COMPARISONS:
            # Store the combined images for the document
            combined = combine_images_with_borders([
                frame / 255,
                gt_fg,
                cv2.cvtColor(ff_fg, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(st_fg, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(mog_fg, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(knn_fg, cv2.COLOR_GRAY2BGR)
            ])
            combined = resize_to_height(combined, 300)

            cv2.imshow("Resultado", combined)
            cv2.waitKey(1)

            # cv2.imwrite(f"../output/docs/static/static_bg_subtraction_{i}.png", combined)

        if GENERATE_GRAPHS:
            if i == 300:
                break

            if i % scores_step != 0:
                continue

            # Calculate scores (precision, recall and f1)
            ff_scores = evaluate_background_subtraction(gt_fg_gray, ff_fg)
            precisions[0].append(ff_scores["precision"])
            recalls[0].append(ff_scores["recall"])
            f1_scores[0].append(ff_scores["f1_score"])

            st_scores = evaluate_background_subtraction(gt_fg_gray, st_fg)
            precisions[1].append(st_scores["precision"])
            recalls[1].append(st_scores["recall"])
            f1_scores[1].append(st_scores["f1_score"])

            mog_scores = evaluate_background_subtraction(gt_fg_gray, mog_fg)
            precisions[2].append(mog_scores["precision"])
            recalls[2].append(mog_scores["recall"])
            f1_scores[2].append(mog_scores["f1_score"])

            knn_scores = evaluate_background_subtraction(gt_fg_gray, knn_fg)
            precisions[3].append(knn_scores["precision"])
            recalls[3].append(knn_scores["recall"])
            f1_scores[3].append(knn_scores["f1_score"])

            print(f"Calculated scores for frame {i}")

    if GENERATE_GRAPHS:
        # Generate and plot results graph
        frame_indices = list(range(0, frames_count - 1, scores_step))

        plt.figure()
        plt.plot(frame_indices, precisions[0], marker="o", markersize=3, label="Static: First frame")
        plt.plot(frame_indices, precisions[1], marker="o", markersize=3, label="Static: Stitched")
        plt.plot(frame_indices, precisions[2], marker="o", markersize=3, label="MoG")
        plt.plot(frame_indices, precisions[3], marker="o", markersize=3, label="KNN")

        plt.title('Precisions')
        plt.xlabel('Frame Index')
        plt.ylabel('Precision')
        plt.legend()

        plt.show()

        # Recall
        plt.figure()
        plt.plot(frame_indices, recalls[0], marker="o", markersize=3, label="Static: First frame")
        plt.plot(frame_indices, recalls[1], marker="o", markersize=3, label="Static: Stitched")
        plt.plot(frame_indices, recalls[2], marker="o", markersize=3, label="MoG")
        plt.plot(frame_indices, recalls[3], marker="o", markersize=3, label="KNN")

        plt.title('Recalls')
        plt.xlabel('Frame Index')
        plt.ylabel('Recall')
        plt.legend()

        plt.show()

        # F1 score
        plt.figure()
        plt.plot(frame_indices, f1_scores[0], marker="o", markersize=3, label="Static: First frame")
        plt.plot(frame_indices, f1_scores[1], marker="o", markersize=3, label="Static: Stitched")
        plt.plot(frame_indices, f1_scores[2], marker="o", markersize=3, label="MoG")
        plt.plot(frame_indices, f1_scores[3], marker="o", markersize=3, label="KNN")

        plt.title('F1 scores')
        plt.xlabel('Frame Index')
        plt.ylabel('F1 score')
        plt.legend()

        plt.show()


if __name__ == "__main__":
    main()
