import cv2
import numpy as np


class ParticleFilter:
    def __init__(self, num_particles, state_dim, motion_model, sensor_model, resample_threshold):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.uniform(-1, 1, (num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles
        self.motion_model = motion_model
        self.sensor_model = sensor_model
        self.resample_threshold = resample_threshold

    def predict(self, velocity):
        self.particles = self.motion_model(self.particles, velocity)

    def update(self, observation):
        for i in range(self.num_particles):
            self.weights[i] = self.sensor_model(self.particles[i], observation)

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= np.sum(self.weights)  # normalize

    def resample(self):
        effective_n = 1. / np.sum(np.square(self.weights))

        if effective_n < self.resample_threshold:
            indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)


def gaussian_sensor_model(particle, observation, sensor_noise_std=0.01):
    error = np.linalg.norm(particle - observation)

    # Force high error particles to disappear
    if error > 1:
        return 0

    return np.exp(-error**2 / (2 * sensor_noise_std**2)) / (sensor_noise_std * np.sqrt(2 * np.pi))


def linear_motion_model(particles, velocity, noise_std=0.01):
    noise = np.random.normal(0, noise_std, particles.shape)
    return particles + velocity + noise


def disperse_motion_model(particles, velocity, noise_std=0.008):
    noise = np.random.normal(0, noise_std, particles.shape)
    return particles + noise  # Ignore velocity
