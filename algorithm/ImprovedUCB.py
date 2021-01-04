import numpy as np
from random import choice

from .IndexAlgorithm import IndexAlgorithm


class ImprovedUCB(IndexAlgorithm):
    """ Ref:
            UCB revisited: Improved regret bounds for the stochastic multi-armed bandit problem.
            Auer, P., & Ortner, R. (2010).
    """
    def __init__(self, nb_arms, horizon, c=1.):
        self.nb_arms = nb_arms
        self.horizon = horizon
        self.c = c

    def start_game(self):
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.delta = 1
        self.batch = list(range(self.nb_arms))
        self.n = 1

    def choice(self):
        if len(self.batch) == 1:
            return self.batch[0]
        else:
            arms_to_explore = [arm for arm in self.batch if self.nb_draws[arm] < self.n]
            if len(arms_to_explore) == 0:
                means = self.cum_reward / self.nb_draws
                confidences = self.c * np.sqrt(2 * np.log(self.horizon * self.delta**2) / self.n)

                ucb = means + confidences
                lcb = means - confidences
                max_lcb = np.amax(lcb)
                new_active_arms = [arm for arm in self.batch if ucb[arm] >= max_lcb]
                self.batch = new_active_arms

                self.delta /= 2.
                self.n = int(2 * np.log(self.horizon * self.delta**2) / self.delta**2) + 1
                return self.choice()
            else:
                return choice(arms_to_explore)
