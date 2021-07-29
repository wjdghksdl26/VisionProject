from pykalman import KalmanFilter
import numpy as np


class Kfilter():
    def __init__(self):
        self.initial_state_mean = None
        self.transition_mat = [[1, 1, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 1],
                               [0, 0, 0, 1]]
        self.observation_mat = [[1, 0, 0, 0],
                                [0, 0, 1, 0]]
        self.kf3 = None
        self.x_now = None
        self.P_now = None

    def trainKfilter(self, pts):
        pts = np.asarray(pts)
        self.initial_state_mean = [pts[0, 0], 0, pts[0, 1], 0]
        kf1 = KalmanFilter(transition_matrices = self.transition_mat,
                           observation_matrices = self.observation_mat,
                           initial_state_mean = self.initial_state_mean)
        kf1 = kf1.em(pts, n_iter=5)
        self.kf3 = KalmanFilter(transition_matrices = self.transition_mat,
                                observation_matrices = self.observation_mat,
                                initial_state_mean = self.initial_state_mean,
                                observation_covariance = 100 * kf1.observation_covariance,
                                em_vars=['transition_covariance', 'initial_state_covariance'])
        self.kf3 = self.kf3.em(pts, n_iter=5)
        (filtered_state_means, filtered_state_covariances) = self.kf3.filter(pts)
        self.x_now = filtered_state_means[-1, :]
        self.P_now = filtered_state_covariances[-1, :]

    def updateKfilter(self, lastpt):
        lastpt = np.asarray(lastpt)
        (self.x_now, self.P_now) = self.kf3.filter_update(filtered_state_mean = self.x_now,
                                                          filtered_state_covariance = self.P_now,
                                                          observation = lastpt)
        
        return (self.x_now[0], self.x_now[2])

#a=Kfilter()
#measurements = np.asarray([(164, 171), (166, 168), (166, 165), (169, 163), (171, 159), (176, 157), (170, 153), (173, 151), (176, 148), (168, 143)])
#a.trainKfilter(measurements)
#new=(171, 142)
#x=a.updateKfilter(new)
#print(x)