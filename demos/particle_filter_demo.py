import cv2

from src.background_subtraction.mog import MogBackgroundSubtraction
from src.techniques.particle_filter_v2 import ParticleFilterV2, disperse_motion_model
from src.techniques.particle_filter_wrapper import ParticleFilterWrapper
from src.utils.background_subtraction import clean_image, fill_contours, remove_shadows
from src.utils.video import get_next_frame


VIDEO_PATH = "../resources/video/Walking.60457274.mp4"


def main():
    # Initialize ParticleFilter
    num_particles = 1000
    pf = ParticleFilterV2(num_particles, disperse_motion_model)

    # Initialize background subtractor
    bg_subtraction = MogBackgroundSubtraction()

    # Initialize the ParticleFilterWrapper
    pfw = ParticleFilterWrapper(pf)

    for frame, fps in get_next_frame(VIDEO_PATH):
        # Apply the background subtractor
        fg = bg_subtraction.apply(frame)

        # Reduce noise
        fg = clean_image(remove_shadows(fg))

        # Fill empty contours to improve the result
        fg = fill_contours(fg)

        pfw.apply(fg)

        pfw.draw(frame, draw_particles=True, draw_estimate=True)

        cv2.imshow("asd", frame)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
