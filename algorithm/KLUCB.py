from math import log

from .IndexAlgorithm import IndexAlgorithm
from .kullback import klucb_bern, klucb_gauss


class KLUCB(IndexAlgorithm):
    """ Ref:
            The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond A. Garivier, O. Cappé
    """
    
    def __init__(self, nb_arms, klucb=klucb_bern):
        self.nb_arms = nb_arms
        self.klucb = klucb

    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float('inf')
        else:
            return self.klucb(self.cum_reward[arm] / self.nb_draws[arm], log(self.t) / self.nb_draws[arm], 1e-4)


class KLUCBPlus(KLUCB):
    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float('inf')
        else:
            return self.klucb(self.cum_reward[arm] / self.nb_draws[arm],
                              log(self.t / self.nb_draws[arm]) / self.nb_draws[arm], 1e-4)

