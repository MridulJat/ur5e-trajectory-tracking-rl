import numpy as np


# Figure-eight (lemniscate) path in XY plane at fixed Z
class FigureEightTrajectory:

    def __init__(self, centre=(0.496, 0.134, 0.579), scale_x=0.10, scale_y=0.07, period=8.0):
        self.centre = np.array(centre, dtype=np.float32)
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.period = period

    def get_position(self, t):
        theta = 2.0 * np.pi * t / self.period

        x = self.centre[0] + self.scale_x * np.sin(theta)
        y = self.centre[1] + self.scale_y * np.sin(2.0 * theta)  # sin(2θ) gives the figure-eight
        z = self.centre[2]

        return np.array([x, y, z], dtype=np.float32)

    def get_velocity(self, t):
        # Analytical derivative of get_position
        theta = 2.0 * np.pi * t / self.period
        dtheta_dt = 2.0 * np.pi / self.period

        vx = self.scale_x * dtheta_dt * np.cos(theta)
        vy = self.scale_y * 2.0 * dtheta_dt * np.cos(2.0 * theta)
        vz = 0.0

        return np.array([vx, vy, vz], dtype=np.float32)
