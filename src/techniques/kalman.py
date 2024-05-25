import numpy as np


class Kalman:
    def __init__(self, dt: float, u: float, std_acc: float, std_meas: float):
        self.a = np.array([
            [1, dt],
            [0, 1],
        ])
        self.b = np.array([
            [(dt ** 2) / 2],
            [dt]
        ])
        self.h = np.array([
            [1, 0],
        ])
        self.u = u
        self.q = np.array([
            [(dt ** 4) / 4, (dt ** 3) / 2],
            [(dt ** 3) / 2, dt ** 2],
        ]) * std_acc ** 2
        self.r = std_meas ** 2
        self.p = np.eye(self.a.shape[1])
        self.x = np.array([
            [0, 500],
            [0, 0],
        ])

    def predict(self):
        self.x = np.dot(self.a, self.x) + np.dot(self.b, self.u)
        self.p = np.dot(np.dot(self.a, self.p), self.a.T) + self.q
        return self.x[0, :]

    def update(self, z):
        S = np.dot(self.h, np.dot(self.p, self.h.T)) + self.r

        K = np.dot(np.dot(self.p, self.h.T), np.linalg.inv(S))
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.h, self.x))))

        identity = np.eye(self.h.shape[1])
        self.p = (identity - (K * self.h)) * self.p
