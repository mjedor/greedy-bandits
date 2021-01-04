import numpy as np

from .IndexAlgorithm import IndexAlgorithm


class AG(IndexAlgorithm):
    def __init__(self, nb_arms, c=1):
        self.nb_arms = nb_arms
        self.c = c

    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float('inf')
        else:
            return self.cum_reward[arm] / self.nb_draws[arm]
        
    def choice(self):
        if self.t <= self.nb_arms:
            return self.t - 1

        index = [self.compute_index(arm) for arm in range(self.nb_arms)]
        ind_max = np.random.choice(np.flatnonzero(index == np.amax(index)))
        
        if np.random.rand() <= min(1, self.c * index[ind_max]):
            return ind_max
        else:
            return np.random.choice(self.nb_arms)
