import numpy as np
from math import ceil, log


def epsilon_t(t):
    return 2 * log(10 * log(t))
    #return log(t)


class UCBV:
    def __init__(self, nb_arms):
        self.nb_arms = nb_arms

    def start_game(self):
        self.t = 1
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.cum_reward2 = np.zeros(self.nb_arms)

    def choice(self):
        if self.t <= 2 * self.nb_arms:
            return (self.t - 1) % self.nb_arms
        
        m = self.cum_reward / self.nb_draws
        v = self.cum_reward2 / self.nb_draws - m*m
        e_t = epsilon_t(self.t)
        index = (m + np.sqrt(2*e_t*v/self.nb_draws) + 3*e_t/self.nb_draws)
        return np.random.choice(np.flatnonzero(index == np.amax(index)))
        
    def get_reward(self, arm, reward):
        self.nb_draws[arm] += 1
        self.cum_reward[arm] += reward
        self.cum_reward2[arm] += reward**2
        self.t += 1

        
class UCBF(UCBV):
    def __init__(self, horizon, beta=1):
        if beta >= 1:
            self.nb_arms = min(ceil(horizon**(beta/(beta+1))), horizon)
        else:
            self.nb_arms = min(ceil(horizon**(beta/2)), horizon)
        self.true_nb_arms = horizon
        
    def start_game(self):
        self.index = np.random.choice(self.true_nb_arms, size=self.nb_arms, replace=False)
        super().start_game()
        
    def choice(self):
        choice = super().choice()
        return self.index[choice]
        
    def get_reward(self, arm, reward):
        sub_index = np.flatnonzero(self.index == arm)[0]
        super().get_reward(sub_index, reward)
