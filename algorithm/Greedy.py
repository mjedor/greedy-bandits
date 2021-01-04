import numpy as np
from random import choice

from .IndexAlgorithm import IndexAlgorithm


class Greedy(IndexAlgorithm):
    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float("inf")
        
        return self.cum_reward[arm] / self.nb_draws[arm]

    
class CAB_Greedy:
    """ Greedy algorithm for continuous-armed bandit problems. """
    
    def __init__(self, nb_arms):
        self.nb_arms = nb_arms

        self.arms = np.arange(1, nb_arms+1) / nb_arms

    def start_game(self):
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)

    def choice(self):
        if self.t <= self.nb_arms:
            self.current_index = self.t - 1
        else:
            index = self.cum_reward / self.nb_draws
            self.current_index = choice(np.flatnonzero(index == np.amax(index)))
        return self.arms[self.current_index]

    def get_reward(self, arm, reward):
        self.nb_draws[self.current_index] += 1
        self.cum_reward[self.current_index] += reward
        self.t += 1
        
        
class SubSampledGreedy:
    """ Greedy algorithm on a subsampling of arms. """
    
    def __init__(self, nb_arms, m):
        self.nb_arms = min(nb_arms, m)
        self.true_nb_arms = nb_arms
        self.m = m
        
    def start_game(self):
        self.index = np.random.choice(self.true_nb_arms, size=self.nb_arms, replace=False)
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        
    def choice(self):
        if self.t <= self.nb_arms:
            return self.index[self.t - 1]
        
        index = self.cum_reward / self.nb_draws
        choice = np.random.choice(np.flatnonzero(index == np.amax(index)))
        return self.index[choice]

    def get_reward(self, arm, reward):
        sub_index = np.flatnonzero(self.index == arm)[0]
        self.nb_draws[sub_index] += 1
        self.cum_reward[sub_index] += reward
        self.t += 1
