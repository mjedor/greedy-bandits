from math import sqrt, log
import numpy as np

from .IndexAlgorithm import IndexAlgorithm


class UCBV(IndexAlgorithm):
    """ The UCB-V algorithm.

    Ref:
        Exploration-exploitation trade-off using variance estimates in multi-armed bandits
        J.-Y. Audibert, R. Munos, Cs. Szepesvári
    """

    def start_game(self):
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.cum_reward2 = np.zeros(self.nb_arms)

    def compute_index(self, arm):
        if self.nb_draws[arm] < 2:
            return float("inf")
        else:
            m = self.cum_reward[arm] / self.nb_draws[arm]
            v = self.cum_reward2[arm] / self.nb_draws[arm] - m*m
            return m + sqrt(2*log(self.t) * v / self.nb_draws[arm]) + 3*log(self.t)/self.nb_draws[arm]

    def get_reward(self, arm, reward):
        self.nb_draws[arm] += 1
        self.cum_reward[arm] += reward
        self.cum_reward2[arm] += reward**2
        self.t += 1
