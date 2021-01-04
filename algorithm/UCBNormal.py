from math import sqrt, log, ceil

from .IndexPolicy import IndexPolicy


class UCBNormal(IndexPolicy):
    """ Ref:
            Finite-time analysis of the multiarmed bandit problem Peter Auer, Nicolò Cesa-Bianchi and Paul Fischer.
    """
    def __init__(self, nb_arms):
        self.nb_arms = nb_arms

    def start_game(self):
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.cum_reward2 = np.zeros(self.nb_arms)

    def compute_index(self, arm):
        if self.nb_draws[arm] < 2 or self.nb_draws[arm] < ceil(8 * log(self.t)):
            return float('inf')
        else:
            return self.cum_reward[arm] / self.nb_draws[arm] \
                   + sqrt(16
                          * (self.cum_reward2[arm] - self.cum_reward[arm]**2/self.nb_draws[arm]) / (self.nb_draws[arm] - 1)
                          * log(self.t-1)/self.nb_draws[arm])

    def get_reward(self, arm, reward):
        self.nb_draws[arm] += 1
        self.cum_reward[arm] += reward
        self.cum_reward2[arm] += reward**2
        self.t += 1
