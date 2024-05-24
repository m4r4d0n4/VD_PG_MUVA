import cv2

from src.background_subtraction.mog import MogBackgroundSubtraction
from src.techniques.particle_filter import disperse_motion_model, gaussian_sensor_model
from src.techniques.particle_filter_v2 import ParticleFilterV2
from src.techniques.particle_filter_wrapper import ParticleFilterWrapper
from src.utils.video import get_next_frame

VIDEO_PATH = "../resources/video/Walking.60457274.mp4"


def main():
    num_particles = 1000
    resample_threshold = num_particles * 0.2
    pf = ParticleFilterV2(num_particles, 2, disperse_motion_model, gaussian_sensor_model, resample_threshold)

    bg_subtraction = MogBackgroundSubtraction()

    pfw = ParticleFilterWrapper(bg_subtraction, pf)

    for frame, fps in get_next_frame(VIDEO_PATH):
        pfw.apply(frame)

        pfw.draw(frame, draw_particles=True, draw_estimate=True, draw_observed=True)

        cv2.imshow("asd", frame)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
