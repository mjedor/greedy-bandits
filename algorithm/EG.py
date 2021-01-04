import numpy as np

from .IndexAlgorithm import IndexAlgorithm


class EG(IndexAlgorithm):
    def __init__(self, nb_arms, epsilon=0.1):
        self.nb_arms = nb_arms
        self.epsilon = epsilon
        
    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float('inf')
        else:
            return self.cum_reward[arm] / self.nb_draws[arm]
        
    def choice(self):
        if self.t <= self.nb_arms:
            return self.t - 1
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.nb_arms)
        else:
            index = [self.compute_index(arm) for arm in range(self.nb_arms)]
            return np.random.choice(np.flatnonzero(index == np.amax(index)))
