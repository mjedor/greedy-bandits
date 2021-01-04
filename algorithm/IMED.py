import numpy as np
from random import choice

from .kullback import kl_bern
from .IndexAlgorithm import IndexAlgorithm


class IMED(IndexAlgorithm):
    """
        Ref: Honda, J., & Takemura, A. (2015). Non-asymptotic analysis of a new bandit algorithm for semi-bounded rewards.
    """
    def __init__(self, nb_arms, kl=kl_bern):
        self.nb_arms = nb_arms
        self.kl = kl

    def choice(self):
        if self.t <= self.nb_arms:
            return self.t - 1

        means = self.cum_reward / self.nb_draws
        best_mean = np.max(means)

        index = np.zeros(self.nb_arms)
        for arm in range(self.nb_arms):
            index[arm] = self.nb_draws[arm] * self.kl(means[arm], best_mean) + np.log(self.nb_draws[arm])

        return choice(np.flatnonzero(index == min(index)))
