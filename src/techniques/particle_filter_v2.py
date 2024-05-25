import numpy as np


class ParticleFilterV2:
    def __init__(self, num_particles, motion_model):
        self.num_particles = num_particles
        self.particles = np.random.uniform(-1, 1, (num_particles, 2))
        self.weights = np.ones(num_particles) / num_particles
        self.motion_model = motion_model

    def predict(self):
        self.particles = self.motion_model(self.particles)

    def update(self, fg_frame):
        h, w = fg_frame.shape[:2]
        half_patch_size = 5

        for i, particle in enumerate(self.particles):
            x, y = int(particle[0] * w), int(particle[1] * h)

            if 0 <= x < w and 0 <= y < h:
                # Extract a small patch around the particle
                patch = fg_frame[
                    max(y - half_patch_size, 0): min(y + half_patch_size, h),
                    max(x - half_patch_size, 0): min(x + half_patch_size, w)
                ]

                self.weights[i] = np.sum(patch)
            else:
                self.weights[i] = 0

        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        # Resample the particles using the "Algoritmo de la ruleta"
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        # Estimate the bounding box based on the min, max particles position
        min_x = np.min(self.particles[:, 0])
        max_x = np.max(self.particles[:, 0])
        min_y = np.min(self.particles[:, 1])
        max_y = np.max(self.particles[:, 1])

        return min_x, min_y, max_x, max_y


def disperse_motion_model(particles, noise_std=0.08):
    noise = np.random.normal(0, noise_std, particles.shape)
    return particles + noise  # Ignore velocity
