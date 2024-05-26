import cv2
import numpy as np


class LucasKanadeTracker:
    def __init__(self):
        
        # Parámetros para el algoritmo de Lucas-Kanade
        self.lk_params = dict(winSize=(35, 35),
                              maxLevel=6,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Crear algunos colores aleatorios para visualizar el seguimiento
        self.color = np.random.randint(0, 255, (100, 3))
        
        # Variables para almacenar el estado anterior
        self.old_gray = None
        self.p0 = None
        self.mask = None

    def initialize(self, frame, mask):
        # Convertir el primer cuadro a escala de grises
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Encuentra las coordenadas (x, y) de los píxeles no nulos en la máscara
        points = np.argwhere(mask != 0)[:, ::-1]
        # Convertir a formato (N, 1, 2) que se espera para los puntos de interés
        points = points.reshape(-1, 1, 2).astype(np.float32)
        num_points = min(1000, len(points))
        indices = np.random.choice(len(points), num_points, replace=False)
        self.p0 = points[indices]
        # Crear una máscara de imagen para dibujar (inicialmente negra)
        self.mask = np.zeros_like(frame)

    def apply(self, frame):
        # Asegurarse de que el tracker está inicializado
        if self.old_gray is None or self.p0 is None:
            raise ValueError("El tracker no está inicializado. Llama a 'initialize' con el primer cuadro.")
        
        # Convertir el nuevo frame a escala de grises
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular el flujo óptico usando Lucas-Kanade
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        
        # Filtrar puntos buenos
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]


            
        # Dibujar las trazas
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a = int(a)
            b = int(b)
            c = int(c)
            d = int(d)
            self.mask = cv2.line(self.mask, (a, b), (c, d), self.color[i % 100].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, self.color[i % 100].tolist(), -1)
        
        # Actualizar el frame anterior y los puntos
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)
        
        # Combinar el frame actual con la máscara
        img = cv2.add(frame, self.mask)

        return img

    def apply_bbox(self,frame):

        # Asegurarse de que el tracker está inicializado
        if self.old_gray is None or self.p0 is None:
            raise ValueError("El tracker no está inicializado. Llama a 'initialize' con el primer cuadro.")
        
        # Convertir el nuevo frame a escala de grises
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular el flujo óptico usando Lucas-Kanade
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        
        # Filtrar puntos buenos
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

        # Calcular la bbox que encierra los puntos
        if len(good_new) > 0:
            min_x = int(np.min(good_new[:, 0]))
            max_x = int(np.max(good_new[:, 0]))
            min_y = int(np.min(good_new[:, 1]))
            max_y = int(np.max(good_new[:, 1]))
            # Dibujar la bbox en el frame
            frame = cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        
        # Actualizar el frame anterior y los puntos
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)
        
        # Combinar el frame actual con la máscara
        img = cv2.add(frame, self.mask)

        return img

