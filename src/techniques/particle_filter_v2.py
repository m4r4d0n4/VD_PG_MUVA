import cv2
import numpy as np


class ParticleFilterV2:
    def __init__(self, num_particles, state_dim, motion_model):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.uniform(-1, 1, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles
        self.motion_model = motion_model
        self.good_particles = np.array([])

    def predict(self):
        self.particles = self.motion_model(self.particles)

    def update(self, fg_frame):
        h, w = fg_frame.shape[:2]
        half_patch_size = 5

        for i, particle in enumerate(self.particles):
            x, y = int(particle[0] * w), int(particle[1] * h)

            if x >= 0 and y >= 0 and x < w and y < h:
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
        for i, weight in enumerate(self.weights):
            x, y = self.particles[i]

            if weight == 0 or not (0 < x < 1) or not (0 < y < 1):
                self.particles[i, 0] = np.random.uniform(0, 1)
                self.particles[i, 1] = np.random.uniform(0, 1)
                self.weights[i] = 0

    def estimate(self, min_weight=0.01):
        self.good_particles = self.particles[self.weights > min_weight]

        if len(self.good_particles) > 0:
            min_x = np.min(self.good_particles[:, 0])
            max_x = np.max(self.good_particles[:, 0])
            min_y = np.min(self.good_particles[:, 1])
            max_y = np.max(self.good_particles[:, 1])

            return min_x, min_y, max_x, max_y

        return None, None, None, None


def gaussian_sensor_model(particle, observation, sensor_noise_std=0.05):
    error = np.linalg.norm(particle - observation)
    likelihood = np.exp(-error**2 / (2 * sensor_noise_std**2)) / (sensor_noise_std * np.sqrt(2 * np.pi))
    return likelihood


def disperse_motion_model(particles, noise_std=0.008):
    noise = np.random.normal(0, noise_std, particles.shape)
    return particles + noise  # Ignore velocity
