from pykalman import KalmanFilter
import numpy as np


def kalman_filter(pts):
    pts = np.asarray(pts)
    initial_state_mean = [pts[0, 0], 0, pts[0, 1], 0]
    transition_mat = [[1, 1, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 0, 1]]
    observation_mat = [[1, 0, 0, 0],
                       [0, 0, 1, 0]]

    kf = KalmanFilter(transition_matrices = transition_mat,
                      observation_matrices = observation_mat,
                      initial_state_mean = initial_state_mean)

    kf = kf.em(pts, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf.filter(pts)
    x = smoothed_state_means[:, 0]
    y = smoothed_state_means[:, 2]

    return x, y