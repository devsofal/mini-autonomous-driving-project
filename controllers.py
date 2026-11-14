import numpy as np
from scipy.linalg import solve_continuous_are


def pure_pursuit(state, road, s, L, lookahead):
    target = road.xy(min(s + lookahead, road.s[-1]))
    dx = target[0] - state[0]
    dy = target[1] - state[1]
    y_body = -np.sin(state[2]) * dx + np.cos(state[2]) * dy
    distance_squared = max(dx * dx + dy * dy, 1e-12)
    return np.arctan2(2 * L * y_body, distance_squared)


def lqr_gain(L, q_y, q_psi, r):
    v = 1.0  # Не влияет на ур. Риккати.
    A = np.array([[0, v], [0, 0]])
    B = np.array([[0], [v / L]])
    Q = np.diag([q_y, q_psi])
    R = np.array([[r]])
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P).ravel()


def lqr(e_y, e_psi, curvature, L, K):
    feedforward = np.arctan(L * curvature)
    feedback = -K[0] * e_y - K[1] * e_psi
    return feedforward + feedback
