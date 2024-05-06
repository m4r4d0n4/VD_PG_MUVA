import cv2


def get_next_frame(path: str):
    vid_capture = cv2.VideoCapture(path)
    fps = vid_capture.get(cv2.CAP_PROP_FPS)

    while vid_capture.isOpened():
        ret, frame = vid_capture.read()

        if not ret:
            break

        yield frame, fps
