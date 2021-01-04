import numpy as np
from numpy.random import choice
from copy import deepcopy

from .kullback import kl_gauss, kl_bern


class OSSB:
    """ The OSSB algorithm for standard multi-armed bandit problems
    
        Ref: Minimal exploration in structured stochastic bandits. Combes, R., Magureanu, S., & Proutiere, A. (2017).

    """
    def __init__(self, nb_arms, kl=kl_bern, epsilon=0., gamma=0.):
        self.nb_arms = nb_arms
        self.kl = kl
        self.epsilon = epsilon
        self.gamma = gamma

    def start_game(self):
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.means = np.zeros(self.nb_arms)
        self.s = 0

    def _solution_optimization_problem(self):
        c = np.zeros(self.nb_arms)

        best_arms = np.flatnonzero(self.means == np.amax(self.means))

        for arm in range(self.nb_arms):
            if arm in best_arms:
                pass
            else:
                c[arm] = 1. / self.kl(self.means[best_arms[0]], self.means[arm])

        return c

    def choice(self):
        if self.t <= self.nb_arms:
            return self.t - 1

        # Solve optimization problem
        c = self._solution_optimization_problem()

        # Exploitation
        if np.all(self.nb_draws >= (1. + self.gamma) * np.log(self.t / self.nb_draws) * c):
            # Choose arm with the best empirical mean
            return choice(np.flatnonzero(self.means == np.amax(self.means)))
        else:
            self.s += 1
            overline_x = self.nb_draws / (c + 1e-15)
            argmin_overline_x = choice(np.flatnonzero(overline_x == np.amin(overline_x)))
            argmin_underline_x = choice(np.flatnonzero(self.nb_draws == np.amin(self.nb_draws)))

            # Estimation
            if self.nb_draws[argmin_underline_x] <= self.epsilon * self.s:
                return argmin_underline_x
            else:  # Exploration
                return argmin_overline_x

    def get_reward(self, arm, reward):
        self.means[arm] = (reward + self.means[arm] * self.nb_draws[arm]) / (self.nb_draws[arm] + 1)
        self.nb_draws[arm] += 1
        self.t += 1
