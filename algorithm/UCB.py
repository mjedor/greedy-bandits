from math import sqrt, log

from .IndexAlgorithm import IndexAlgorithm


class UCB(IndexAlgorithm):
    """ Ref:
            Finite-time analysis of the multi-armed bandit problem Peter Auer, Nicolò Cesa-Bianchi and Paul Fischer.
    """
    def __init__(self, nb_arms, c=1.):
        self.nb_arms = nb_arms
        self.c = c

    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float('inf')
        else:
            return self.cum_reward[arm] / self.nb_draws[arm] + self.c * sqrt(2 * log(self.t) / self.nb_draws[arm])
