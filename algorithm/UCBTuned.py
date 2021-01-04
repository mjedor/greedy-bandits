from math import sqrt, log
import numpy as np

from .IndexAlgorithm import IndexAlgorithm


class UCBTuned(IndexAlgorithm):
    """ Ref:
            Finite-time analysis of the multiarmed bandit problem Peter Auer, Nicolò Cesa-Bianchi and Paul Fischer.
    """

    def start_game(self):
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.cum_reward2 = np.zeros(self.nb_arms)

    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float('inf')
        else:
            m = self.cum_reward[arm] / self.nb_draws[arm]
            v = self.cum_reward2[arm] / self.nb_draws[arm] - m * m + sqrt(2*log(self.t) / self.nb_draws[arm])
            return m + sqrt(log(self.t) / self.nb_draws[arm] * min(1/4, v))

    def get_reward(self, arm, reward):
        self.nb_draws[arm] += 1
        self.cum_reward[arm] += reward
        self.cum_reward2[arm] += reward ** 2
        self.t += 1
