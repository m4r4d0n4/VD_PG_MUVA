import cv2

from src.utils.video import get_next_frame
from src.techniques.yolo_person_detector import YOLOPersonDetector
import time
import statistics


VIDEO_PATH = "./resources/video/Walking.60457274.mp4"
MOSTRAR_IMAGENES = True


def main():
    # YOLO for detecting people
    detector = YOLOPersonDetector()
    
    start_time = time.time()
    tiempos = []

    for frame, fps in get_next_frame(VIDEO_PATH):
        # Detecting people in the image
        frame, center_x,center_y, x1, y1, x2, y2, conf = detector.apply(frame)
        # Convert to ms
        total_time = (time.time() - start_time) * 1000

        tiempos.append(total_time)
        start_time = time.time()
        if MOSTRAR_IMAGENES:
            cv2.imshow('Person Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    
    media = statistics.mean(tiempos)

    desviacion_estandar = statistics.stdev(tiempos)
    print("Media:", media)
    print("Desviación estándar:", desviacion_estandar)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
