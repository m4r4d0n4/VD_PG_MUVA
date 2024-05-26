import cv2

from src.utils.video import get_next_frame, create_video_as_other
from src.techniques.yolo_person_detector import YOLOPersonDetector


VIDEO_PATH = "../resources/video/Walking.60457274.mp4"


def main():
    # Crear una instancia del detector de personas
    detector = YOLOPersonDetector()

    video = create_video_as_other(VIDEO_PATH, "../output/docs/videos/demo_yolo.mp4")

    for frame, fps in get_next_frame(VIDEO_PATH):
        # Aplicar la detección de personas
        frame, center_x, center_y, x1, y1, x2, y2, conf = detector.apply(frame)

        """
        # Mostrar el frame resultante (opcional)
        cv2.imshow('Person Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        """
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    main()
