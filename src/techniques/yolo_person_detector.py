import torch
import cv2

class YOLOPersonDetector:
    """Clase para detectar personas utilizando YOLOv5.

    Args:
        model_name (str, optional): Nombre del modelo YOLOv5 a utilizar. Puede ser 'yolov5s', 'yolov5m', 'yolov5l' o 'yolov5x'.
            Por defecto, se utiliza 'yolov5s'.
        confidence_threshold (float, optional): Umbral de confianza para filtrar las detecciones. Las detecciones con una
            confianza menor que este valor serán ignoradas. Debe ser un valor entre 0 y 1. Por defecto, se utiliza 0.3.

    Attributes:
        model: Modelo YOLOv5 cargado.
        confidence_threshold (float): Umbral de confianza utilizado para filtrar las detecciones.

    """
    def __init__(self, model_name='yolov5s', confidence_threshold=0.3):
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.confidence_threshold = confidence_threshold

    def apply(self, frame):
        """Aplica la detección de personas en un cuadro de imagen.

        Args:
            frame (array): Imagen (numpy array) en formato BGR.

        Returns:
            tuple: Una tupla que contiene:
                - El cuadro de imagen con las detecciones dibujadas.
                - La coordenada x del centro del rectángulo de detección.
                - Las coordenadas x1, y1, x2, y2 de los vértices del rectángulo de detección.
                - La confianza de la detección.

        """
        frame = frame.copy()
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]
        person_detections = [det for det in detections if det[5] == 0]  # Clase 0 es 'persona'

        for det in person_detections:
            x1, y1, x2, y2, conf, _ = det
            if conf < self.confidence_threshold:
                continue

            # Dibujar la caja delimitadora en el cuadro
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Calcular y dibujar el centro del rectángulo
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        return frame , center_x, center_y , x1, y1, x2, y2, conf
if __name__ == "__main__":
    # Ruta al video de entrada
    video_path = "./resources/video/Walking.60457274.mp4"

    # Captura de video
    cap = cv2.VideoCapture(video_path)

    # Obtener dimensiones del video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Definir el codec y crear el objeto VideoWriter para el video de salida
    #out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

    # Crear una instancia del detector de personas
    detector = YOLOPersonDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Aplicar la detección de personas
        frame, center_x,center_y, x1, y1, x2, y2, conf = detector.apply(frame)


        # Escribir el cuadro en el video de salida
        #out.write(frame)

        # Mostrar el cuadro resultante (opcional)
        cv2.imshow('Person Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los objetos de video
    cap.release()
    #out.release()
    cv2.destroyAllWindows()
