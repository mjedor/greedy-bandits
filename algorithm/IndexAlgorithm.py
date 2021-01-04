from random import choice
import numpy as np


class IndexAlgorithm:
    """ Class that implements a generic index algorithm """

    def __init__(self, nb_arms):
        self.nb_arms = nb_arms
        
    def start_game(self):
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)

    def choice(self):
        """ In an index algorithm, choose at random an arm with maximal index """

        if self.t <= self.nb_arms:
            return self.t - 1

        index = [self.compute_index(arm) for arm in range(self.nb_arms)]
        return choice(np.flatnonzero(index == np.amax(index)))

    def get_reward(self, arm, reward):
        self.nb_draws[arm] += 1
        self.cum_reward[arm] += reward
        self.t += 1
        