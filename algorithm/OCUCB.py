from math import sqrt, log, exp

from .IndexAlgorithm import IndexAlgorithm


class OCUCB(IndexAlgorithm):
    """ Ref:
            Lattimore, T. (2015). Optimally confident UCB: Improved regret for finite-armed bandits.
    """
    def __init__(self, nb_arms, horizon, alpha=3., psi=2.):
        self.nb_arms = nb_arms
        self.horizon = horizon
        self.alpha = alpha
        self.psi = psi

    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float('inf')
        else:
            return self.cum_reward[arm] / self.nb_draws[arm] + \
                   sqrt(self.alpha / self.nb_draws[arm] * log(self.psi * self.horizon / self.t))

        
class OCUCBn(IndexAlgorithm):
    """ Ref:
            Lattimore, T. (2016). Regret analysis of the anytime optimally confident UCB algorithm.
    """
    def __init__(self, nb_arms, eta=2., rho=1.):
        self.nb_arms = nb_arms
        self.eta = eta
        self.rho = rho

    def _Bterm(self, arm):
        # Compute second part of third term
        temp = 0.
        for j in range(self.nb_arms):
            temp += min(self.nb_draws[arm], self.nb_draws[j]**self.rho * self.nb_draws[arm]**(1-self.rho))

        return max(exp(1), log(self.t), self.t * log(self.t) / temp)

    def compute_index(self, arm):
        if self.nb_draws[arm] == 0:
            return float('inf')
        else:
            return self.cum_reward[arm] / self.nb_draws[arm] + \
                   sqrt(2 * self.eta * log(self._Bterm(arm)) / self.nb_draws[arm])