import numpy as np

from .kullback import kl_bern
from .IndexAlgorithm import IndexAlgorithm


class DMED(IndexAlgorithm):
    """ DMED algorithm  
        Ref: Honda, J., & Takemura, A. (2010, June). An Asymptotically Optimal Bandit Algorithm for Bounded Support Models.
    """
    def __init__(self, nb_arms, kl=kl_bern):
        self.nb_arms = nb_arms
        self.kl = np.vectorize(kl)

    def start_game(self):
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.next_actions = list(range(self.nb_arms))

    def compute_next_actions(self, means, best_mean):
        self.next_actions = np.flatnonzero(self.nb_draws * self.kl(means, best_mean) < np.log(self.t))
        self.next_actions = list(self.next_actions)
        
    def choice(self):
        if len(self.next_actions) == 0:
            means = self.cum_reward / self.nb_draws
            best_mean = np.max(means)
            self.compute_next_actions(means, best_mean)

        # Play next action
        return self.next_actions.pop(0)


class DMEDPlus(DMED):
    def compute_next_actions(self, means, best_mean):
        self.next_actions = np.flatnonzero(self.nb_draws * self.kl(means, best_mean) < np.log(self.t / self.nb_draws))
        self.next_actions = list(self.next_actions)
