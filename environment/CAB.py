import numpy as np
from math import sqrt
from random import gauss


class ResultCAB:
    def __init__(self, horizon):
        self.choices = np.zeros(horizon)
        self.rewards = np.zeros(horizon)

    def store(self, t, choice, reward):
        self.choices[t] = choice
        self.rewards[t] = reward

    
class CAB:
    def __init__(self, f, sigma_2=1):
        self.f = f
        self.sigma_2 = sigma_2

    def play(self, algorithm, horizon):
        result = ResultCAB(horizon)
        algorithm.start_game()
        for t in range(horizon):
            choice = algorithm.choice()
            f_x = self.f(choice)
            reward = f_x + gauss(0, sqrt(self.sigma_2))
            algorithm.get_reward(choice, reward)
            result.store(t, choice, f_x)
    
        return result
    

class EvaluationCAB:
    def __init__(self, env, algorithm, nb_repetitions, horizon, tsav=[]):
        if len(tsav) > 0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(horizon)
            
        self.env = env
        self.cum_reward = np.zeros((nb_repetitions, len(self.tsav)))

        for k in range(nb_repetitions):
            if nb_repetitions < 10 or k % (nb_repetitions / 10) == 0:
                print(k)
            result = env.play(algorithm, horizon)
            self.cum_reward[k, :] = np.cumsum(result.rewards)[self.tsav]

    def std_regret(self):
        max_f = max([self.env.f(x) for x in np.linspace(0, 1, 1001, endpoint=True)])
        oracle = (1 + self.tsav) * max_f
        return np.std(oracle-self.cum_reward, 0) 

    def mean_regret(self):
        max_f = max([self.env.f(x) for x in np.linspace(0, 1, 1001, endpoint=True)])
        return (1 + self.tsav) * max_f - np.mean(self.cum_reward, 0)
    