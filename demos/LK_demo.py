import cv2

from src.background_subtraction.mog import MogBackgroundSubtraction
from src.utils.background_subtraction import clean_image, remove_shadows
from src.utils.video import get_next_frame
from src.techniques.lucas_kanade import LucasKanadeTracker
import time
import statistics


VIDEO_PATH = "./resources/video/Walking.60457274.mp4"
MOSTRAR_IMAGENES = True


def main():
    # LK tracker
    lk = LucasKanadeTracker()
    
    # Initialize background subtractor
    bg_subtraction = MogBackgroundSubtraction()
    init = False
    # Start timer
    start_time = time.time()
    tiempos = []
    for frame, fps in get_next_frame(VIDEO_PATH):
        foreground = bg_subtraction.apply(frame)
        foreground = remove_shadows(foreground)
        foreground = clean_image(foreground)
        #print(cv2.countNonZero(foreground))
        if not init and cv2.countNonZero(foreground) > 10000:
            lk.initialize(frame,foreground)
            if MOSTRAR_IMAGENES:
                cv2.imshow("Lucas-Kanade", foreground)
                cv2.waitKey(0)
            init = True
        if init:
            # Uncomment to see the boxes
            f = lk.apply_bbox(frame)
            # Uncomment to see the OF
            #f = lk.apply(frame)
            # Convert to ms
            total_time = (time.time() - start_time) * 1000

            tiempos.append(total_time)
            start_time = time.time()
            #print("Tiempo total de ejecución:", total_time, "ms")
            if MOSTRAR_IMAGENES:
                cv2.imshow("Lucas-Kanade", f)
                cv2.waitKey(int(fps))
    
    media = statistics.mean(tiempos)
    desviacion_estandar = statistics.stdev(tiempos)
    print("Media:", media)
    print("Desviación estándar:", desviacion_estandar)


if __name__ == "__main__":
    main()
