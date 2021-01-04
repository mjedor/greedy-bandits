from numpy.random import normal
from math import sqrt


class ImproperGaussian:

    def __init__(self):
        self.cum_reward = 0
        self.nb_samples = 0

    def reset(self):
        self.cum_reward = 0
        self.nb_samples = 0

    def update(self, obs):
        self.nb_samples += 1
        self.cum_reward += obs

    def sample(self):
        return normal(self.cum_reward / self.nb_samples, 1 / sqrt(self.nb_samples))
