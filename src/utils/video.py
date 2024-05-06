import cv2


def get_next_frame(path: str):
    """
    Helper function that returns an iterator with next frame of a video (the initial one if called for the first time).
    :param path: The path to load the video from.
    :return: An iterator with the frame.
    """
    vid_capture = cv2.VideoCapture(path)
    fps = vid_capture.get(cv2.CAP_PROP_FPS)

    while vid_capture.isOpened():
        ret, frame = vid_capture.read()

        if not ret:
            break

        yield frame, fps
