import cv2
import sys,os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.background_subtraction.mog import MogBackgroundSubtraction
from src.techniques.particle_filter_v2 import ParticleFilterV2, disperse_motion_model
from src.techniques.particle_filter_wrapper import ParticleFilterWrapper
from src.utils.video import get_next_frame
from src.techniques.yolo_person_detector import YOLOPersonDetector

VIDEO_PATH = "./resources/video/Walking.60457274.mp4"


def main():

    # Crear una instancia del detector de personas
    detector = YOLOPersonDetector()

    for frame, fps in get_next_frame(VIDEO_PATH):
        # Aplicar la detección de personas
        frame, center_x,center_y, x1, y1, x2, y2, conf = detector.apply(frame)

        # Mostrar el frame resultante (opcional)
        cv2.imshow('Person Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
