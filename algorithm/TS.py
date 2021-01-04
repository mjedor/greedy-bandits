from random import choice
import numpy as np

from .IndexAlgorithm import IndexAlgorithm


class TS(IndexAlgorithm):
    """ The Thompson (Bayesian) index algorithm for bounded rewards.
    Ref:
        Analysis of Thompson sampling for the multi-armed bandit problem. S Agrawal, N Goyal
    """

    def __init__(self, nb_arms, posterior):
        self.nb_arms = nb_arms
        self.t = 1
        self.posterior = dict()
        for arm in range(self.nb_arms):
            self.posterior[arm] = posterior()

    def start_game(self):
        self.t = 1
        for arm in range(self.nb_arms):
            self.posterior[arm].reset()

    def choice(self):
        index = [self.compute_index(arm) for arm in range(self.nb_arms)]
        return choice(np.flatnonzero(index == np.amax(index)))        
            
    def get_reward(self, arm, reward):
        self.posterior[arm].update(reward)
        self.t += 1

    def compute_index(self, arm):
        return self.posterior[arm].sample()
    
    
class TSGaussian(TS):
    """ The Thompson (Bayesian) index policy for Gaussian rewards."""
    
    def choice(self):
        if self.t <= self.nb_arms:
            return self.t - 1

        index = [self.compute_index(arm) for arm in range(self.nb_arms)]
        return choice(np.flatnonzero(index == np.amax(index)))
