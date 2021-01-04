import numpy as np

from .IndexAlgorithm import IndexAlgorithm


def overline_log(x):
    return np.log((x + np.exp(1)) * np.sqrt(np.log(x + np.exp(1))))


class AdaUCB(IndexAlgorithm):
    """ Ref:
            Lattimore, T. (2018). Refining the confidence level for optimistic bandit strategies.
    """
    def __init__(self, nb_arms, horizon):
        self.nb_arms = nb_arms
        self.horizon = horizon

    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float('inf')
        else:
            denom = 0
            for arm2 in range(self.nb_arms):
                denom += min(self.nb_draws[arm], np.sqrt(self.nb_draws[arm] * self.nb_draws[arm2]))
            return self.cum_reward[arm] / self.nb_draws[arm] + \
                   np.sqrt(2 / self.nb_draws[arm] * overline_log(self.horizon / denom))
