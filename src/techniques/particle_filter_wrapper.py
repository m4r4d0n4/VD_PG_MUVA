import cv2

from src.techniques.particle_filter_v2 import ParticleFilterV2
from src.utils.background_subtraction import clean_image, remove_shadows


class ParticleFilterWrapper:
    def __init__(self, bg_subtractor, particle_filter: ParticleFilterV2):
        self.pf = particle_filter
        self.bg_subtractor = bg_subtractor

        self.estimated_pos = None, None

    def initialize(self):
        pass

    def apply(self, frame):
        h, w = frame.shape[:2]

        # Calculate foreground/background mask
        fg = self.bg_subtractor.apply(frame)
        fg = clean_image(remove_shadows(fg))

        # Fill empty contours for a better result
        contours, _ = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fg = cv2.drawContours(fg, contours, -1, color=(255), thickness=cv2.FILLED)

        # Calculate particle steps
        self.pf.predict()
        self.pf.update(fg)
        self.pf.resample()

        # Retrieve estimate
        min_x, min_y, max_x, max_y = self.pf.estimate()

        # Recalculate the bounding box in image coordinates
        min_x, max_x = int(min_x * w), int(max_x * w)
        min_y, max_y = int(min_y * h), int(max_y * h)

        self.estimated_pos = min_x, min_y, max_x, max_y

        return self.estimated_pos

    def draw(self, frame, draw_particles=True, draw_estimate=True):
        h, w = frame.shape[:2]

        # If active, draw all the particles
        if draw_particles:
            for i, particle in enumerate(self.pf.particles):
                cv2.circle(frame, (int(particle[0] * w), int(particle[1] * h)), 2, (255, 0, 255), -1)

        # If active, draw the bounding box
        if draw_estimate and self.estimated_pos[0] is not None:
            cv2.rectangle(frame, self.estimated_pos[:2], self.estimated_pos[2:4], (255, 0, 0), 2)
