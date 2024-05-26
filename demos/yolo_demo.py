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
import time
import statistics
VIDEO_PATH = "./resources/video/Walking.60457274.mp4"
MOSTRAR_IMAGENES = True

def main():

    # Crear una instancia del detector de personas
    detector = YOLOPersonDetector()
    # Inicializar el tiempo de inicio
    start_time = time.time()
    tiempos = []

    for frame, fps in get_next_frame(VIDEO_PATH):
        # Aplicar la detección de personas
        frame, center_x,center_y, x1, y1, x2, y2, conf = detector.apply(frame)
        # Convertir el tiempo total de ejecución a milisegundos
        total_time = (time.time() - start_time) * 1000

        tiempos.append(total_time)
        start_time = time.time()
        if MOSTRAR_IMAGENES:
            # Mostrar el frame resultante (opcional)
            cv2.imshow('Person Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Calcular la media
    media = statistics.mean(tiempos)

    # Calcular la desviación estándar
    desviacion_estandar = statistics.stdev(tiempos)
    print("Media:", media)
    print("Desviación estándar:", desviacion_estandar)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
