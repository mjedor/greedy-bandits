from math import sqrt, log
import numpy as np
from random import choice

from .IndexAlgorithm import IndexAlgorithm


class MOSS(IndexAlgorithm):
    """ Ref:
            Minimax Policies for Adversarial and Stochastic Bandits J-Y. Audibert and S. Bubeck
    """
    def __init__(self, nb_arms, horizon, c=4):
        self.nb_arms = nb_arms
        self.horizon = horizon
        self.c = c

    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float('inf')
        else:
            return self.cum_reward[arm] / self.nb_draws[arm] + \
                   sqrt(max(0., self.c * log(self.horizon / (self.nb_arms * self.nb_draws[arm]))) / self.nb_draws[arm])

        
class CAB_MOSS:
    """ MOSS algorithm for continuous-armed bandit problems. """
    
    def __init__(self, nb_arms, horizon, c=4):
        self.nb_arms = nb_arms
        self.horizon = horizon
        self.c = c

        self.arms = np.arange(1, nb_arms+1) / nb_arms

    def start_game(self):
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)

    def choice(self):
        if self.t <= self.nb_arms:
            self.current_index = self.t - 1
        else:
            index = self.cum_reward / self.nb_draws + np.sqrt(self.c * np.log(np.maximum(1, self.horizon / (self.nb_arms * self.nb_draws))) / self.nb_draws)
            self.current_index = choice(np.flatnonzero(index == np.amax(index)))
        return self.arms[self.current_index]

    def get_reward(self, arm, reward):
        self.nb_draws[self.current_index] += 1
        self.cum_reward[self.current_index] += reward
        self.t += 1
        
        
class SubSampledMOSS:
    """ MOSS algorithm on a subsampling of arms. """
    
    def __init__(self, nb_arms, m, horizon, c=4):
        self.true_nb_arms = nb_arms
        self.nb_arms = min(nb_arms, m)
        self.horizon = horizon
        self.c = c

    def start_game(self):
        self.arms = np.random.choice(self.true_nb_arms, size=self.nb_arms, replace=False)
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)

    def choice(self):
        if self.t <= self.nb_arms:
            self.current_index = self.t - 1
        else:
            index = self.cum_reward / self.nb_draws + np.sqrt(self.c * np.log(np.maximum(1, self.horizon / (self.nb_arms * self.nb_draws))) / self.nb_draws)
            self.current_index = np.random.choice(np.flatnonzero(index == np.amax(index)))
        return self.arms[self.current_index]

    def get_reward(self, arm, reward):
        self.nb_draws[self.current_index] += 1
        self.cum_reward[self.current_index] += reward
        self.t += 1