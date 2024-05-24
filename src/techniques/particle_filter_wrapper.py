import cv2
import numpy as np

from src.techniques.particle_filter_v2 import ParticleFilterV2
from src.utils.background_subtraction import clean_image, remove_shadows
from src.utils.get_center_from_foreground import get_center_from_foreground


class ParticleFilterWrapper:
    def __init__(self, bg_subtractor, particle_filter: ParticleFilterV2):
        self.pf = particle_filter
        self.bg_subtractor = bg_subtractor

        self.observed_pos = None, None
        self.estimated_pos = None, None

    def initialize(self):
        pass

    def apply(self, frame):
        h, w = frame.shape[:2]

        fg = self.bg_subtractor.apply(frame)
        fg = clean_image(remove_shadows(fg))

        contours, _ = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fg = cv2.drawContours(fg, contours, -1, color=(255), thickness=cv2.FILLED)

        x, y = get_center_from_foreground(fg)

        if x is None and y is None:
            return None, None

        observation = np.array([x / w, y / h])

        velocity = 0, 0

        if self.observed_pos[0] is not None and observation[0] is not None:
            velocity = np.array(observation) - np.array(self.observed_pos)
            velocity = velocity / max(abs(velocity))
            velocity /= 1000

        self.pf.predict(velocity)
        self.pf.update(fg)
        self.pf.resample()

        # self.estimated_pos = self.pf.estimate()
        min_x, min_y, max_x, max_y = self.pf.estimate()

        if min_x is not None:
            min_x, max_x = int(min_x * w), int(max_x * w)
            min_y, max_y = int(min_y * h), int(max_y * h)

            self.estimated_pos = min_x, min_y, max_x, max_y

        self.observed_pos = x, y

        # print(f"Observed: {int(self.estimated_pos[0] * w), int(self.estimated_pos[1] * h)}")
        # print(f"Estimated: {self.observed_pos}")

        return self.estimated_pos

    def draw(self, frame, draw_particles=True, draw_estimate=True, draw_observed=True):
        h, w = frame.shape[:2]

        if draw_particles:
            for i, particle in enumerate(self.pf.particles):
                weight = int(self.pf.weights[i] * 50)

                if weight != 0:
                    cv2.circle(frame, (int(particle[0] * w), int(particle[1] * h)), weight, (255, 0, 255), -1)

        if draw_estimate and self.estimated_pos[0] is not None:
            # cv2.circle(frame, (int(self.estimated_pos[0]), int(self.estimated_pos[1] * h)), 10, (255, 255, 0))
            cv2.rectangle(frame, self.estimated_pos[:2], self.estimated_pos[2:4], (255, 0, 0), 2)

        if draw_observed and self.observed_pos[0] is not None:
            cv2.circle(frame, (self.observed_pos[0], self.observed_pos[1]), 10, (0, 0, 255))
