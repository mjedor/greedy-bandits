from .IndexAlgorithm import IndexAlgorithm


class BayesUCB(IndexAlgorithm):
    """ The Bayes-UCB algorithm.
    Ref:
        On Bayesian upper confidence bounds for bandit problems. E Kaufmann, O Cappé, A Garivier
    """

    def __init__(self, nb_arms, posterior, power=1):
        self.nb_arms = nb_arms
        self.power = power

        self.t = 1
        self.posterior = dict()
        for arm in range(self.nb_arms):
            self.posterior[arm] = posterior()

    def start_game(self):
        self.t = 1
        for arm in range(self.nb_arms):
            self.posterior[arm].reset()

    def get_reward(self, arm, reward):
        self.posterior[arm].update(reward)
        self.t += 1

    def compute_index(self, arm):
        return self.posterior[arm].quantile(1 - 1. / (self.t ** self.power))
