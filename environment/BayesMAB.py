import numpy as np


class EvaluationBayesMAB:
    """ Evaluation class for a Bayesian multi-armed bandit problem """
    
    def __init__(self, envs, pol, horizon, tsav=[]):
        if len(tsav) > 0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(horizon)
        
        self.envs = envs
        self.nb_repetitions = len(envs)
        self.cum_reward = np.zeros((self.nb_repetitions, len(self.tsav)))
        self.oracle = np.zeros((self.nb_repetitions, len(self.tsav)))

        for k in range(self.nb_repetitions):
            if self.nb_repetitions < 10 or k % (self.nb_repetitions / 10) == 0:
                print(k)
                
            result = envs[k].play(pol, horizon)
            self.cum_reward[k, :] = np.cumsum(result.rewards)[self.tsav]
            self.oracle[k, :] = (1 + self.tsav) * max([arm.expectation for arm in self.envs[k].arms])

    def std_regret(self):
        return np.std(self.oracle - self.cum_reward, 0)

    def mean_regret(self):
        return np.mean(self.oracle, 0) - np.mean(self.cum_reward, 0)
