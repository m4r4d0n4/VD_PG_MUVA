import cv2


def get_next_frame(path: str):
    """
    Helper function that returns an iterator with next frame of a video (the initial one if called for the first time).
    :param path: The path to load the video from.
    :return: An iterator with the frames.
    """
    vid_capture = cv2.VideoCapture(path)
    fps = vid_capture.get(cv2.CAP_PROP_FPS)

    while vid_capture.isOpened():
        ret, frame = vid_capture.read()

        if not ret:
            break

        yield frame, fps


def create_video(path: str, width: int, height: int, frames: float = 30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, frames, (width, height))


def create_video_as_other(input_video_path: str, output_video_path: str):
    vid_capture = cv2.VideoCapture(input_video_path)
    fps = vid_capture.get(cv2.CAP_PROP_FPS)

    ret, frame = vid_capture.read()
    h, w = frame.shape[:2]

    return create_video(output_video_path, w, h, fps)
