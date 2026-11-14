import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


def vehicle_rhs(pose, delta, v, L):
    # pose = [x, y, psi], точка автомобиля — центр задней оси.
    return np.array([v * np.cos(pose[2]), v * np.sin(pose[2]),
                     v * np.tan(delta) / L])


def step(state, command, v, tau, dt, L):
    if tau == 0:
        delta_start = command
        delta_middle = command
        delta_end = command
    else:
        delta_start = state[3]
        delta_middle = command + (state[3] - command) * np.exp(-dt / (2 * tau))
        delta_end = command + (state[3] - command) * np.exp(-dt / tau)

    pose = state[:3]
    k1 = vehicle_rhs(pose, delta_start, v, L)
    k2 = vehicle_rhs(pose + dt * k1 / 2, delta_middle, v, L)
    k3 = vehicle_rhs(pose + dt * k2 / 2, delta_middle, v, L)
    k4 = vehicle_rhs(pose + dt * k3, delta_end, v, L)
    pose_new = pose + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return np.array([pose_new[0], pose_new[1], pose_new[2], delta_end])


class Road:
    def __init__(self, ds=0.05):
        self.s = np.linspace(0, 305, int(round(205 / ds)) + 1)
        self.ds = self.s[1] - self.s[0]
        curvature = np.zeros_like(self.s)
        turns = [
            ( 70, 115,   0.070),
            (130, 175,  0.075),
            (190, 240,  -0.065),
            (255, 295,   0.055),
        ]
        for start, end, amplitude in turns:
            u = np.clip((self.s - start) / (end - start), 0, 1)
            curvature += amplitude * np.sin(np.pi * u) ** 4

        heading = cumulative_trapezoid(curvature, self.s, initial=0)
        self.x = cumulative_trapezoid(np.cos(heading), self.s, initial=0)
        self.y = cumulative_trapezoid(np.sin(heading), self.s, initial=0)
        self.sx = CubicSpline(self.s, self.x)
        self.sy = CubicSpline(self.s, self.y)

    def xy(self, s):
        return np.array([self.sx(s), self.sy(s)])

    def errors(self, state):
        x, y, psi, delta = state

        distances = (self.x - x) ** 2 + (self.y - y) ** 2
        index = np.argmin(distances)
        lower = max(0, self.s[index] - 2 * self.ds)
        upper = min(self.s[-1], self.s[index] + 2 * self.ds)

        def distance(s):
            return (self.sx(s) - x) ** 2 + (self.sy(s) - y) ** 2

        result = minimize_scalar(distance, bounds=(lower, upper), method='bounded',
                                 options={'xatol': 1e-9})
        candidates = [lower, result.x, upper]
        s = candidates[np.argmin([distance(q) for q in candidates])]
        dx, dy = self.sx(s, 1), self.sy(s, 1)
        heading = np.arctan2(dy, dx)
        curvature = (dx * self.sy(s, 2) - dy * self.sx(s, 2)) / (dx * dx + dy * dy) ** 1.5
        e_y = -np.sin(heading) * (x - self.sx(s)) + np.cos(heading) * (y - self.sy(s))
        e_psi = (psi - heading + np.pi) % (2 * np.pi) - np.pi
        return s, e_y, e_psi, curvature
