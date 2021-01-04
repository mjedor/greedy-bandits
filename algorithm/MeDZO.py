import numpy as np
from math import ceil, log2


class MeDZO:
    """
        Ref: Polynomial cost of adaptation for x-armed bandits. Hadiji, H. (2019).
    """
    def __init__(self, horizon, B):
        self.horizon = horizon
        self.p = ceil(log2(B))
        
    def start_game(self):
        self.i = 1
        self.t = 1
        self.nb_arms = 2**(self.p + 2 - self.i)
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.arms = np.arange(1, self.nb_arms+1) / self.nb_arms
        self.previous_arms = [self.arms]
        self.probas = []

    def restart(self):
        self.t = 1
        self.probas += [self.nb_draws / np.sum(self.nb_draws)]
        self.nb_arms = 2**(self.p + 2 - self.i)
        self.nb_draws = np.zeros(self.nb_arms + self.i-1)
        self.cum_reward = np.zeros(self.nb_arms + self.i-1)
        self.arms = np.arange(1, self.nb_arms+1) / self.nb_arms
        self.previous_arms += [self.arms]

    def choice(self):
        if self.t <= self.nb_arms + self.i-1:
            self.temp_choice = self.t - 1
        else:
            index = self.cum_reward / self.nb_draws + np.sqrt(4 * np.log(np.maximum(1, 2**(self.p + self.i) / (self.nb_arms * self.nb_draws))) / self.nb_draws)
            self.temp_choice = np.random.choice(np.flatnonzero(index == np.amax(index)))
        
        if self.temp_choice < self.nb_arms:
            return self.arms[self.temp_choice]
        else:
            j = self.temp_choice - self.nb_arms
            temp_nb_arms = len(self.previous_arms[j]) + j
            temp = np.random.choice(temp_nb_arms, p=self.probas[j])
            if temp < len(self.previous_arms[j]):
                return self.previous_arms[j][temp]
            else:
                for k in range(j-1, 0, -1):
                    temp_nb_arms = len(self.previous_arms[k]) + k
                    temp = np.random.choice(temp_nb_arms, p=self.probas[k])
                    if temp < len(self.previous_arms[j]):
                        return self.previous_arms[k][temp]
                temp = np.random.choice(len(self.previous_arms[0]), p=self.probas[0])
                return self.previous_arms[0][temp]
        
    def get_reward(self, arm, reward):
        self.nb_draws[self.temp_choice] += 1
        self.cum_reward[self.temp_choice] += reward
            
        self.t += 1
        if self.t > 2**(self.p + self.i):
            self.i +=1
            self.restart()
            
            
class empMeDZO(MeDZO):
    """ Empirical version of MeDZO for continuous-armed bandit problems.
        Ref: On Regret with Multiple Best Arms. Zhu, Y., & Nowak, R. (2020).
    """
    def start_game(self):
        self.i = 1
        self.t = 1
        self.nb_arms = 2**(self.p + 2 - self.i)
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.arms = np.arange(1, self.nb_arms+1) / self.nb_arms

    def restart(self):
        self.t = 1
        self.nb_arms = 2**(self.p + 2 - self.i)
        
        index = self.cum_reward / self.nb_draws
        ind = np.argsort(-index)[:self.nb_arms]
        self.arms = self.arms[ind]
        self.cum_reward = self.cum_reward[ind]
        self.nb_draws = self.nb_draws[ind]
    
    def choice(self):
        if self.i == 1 and self.t <= self.nb_arms:
            self.temp_choice = self.t - 1
            return self.arms[self.t - 1]
        
        index = self.cum_reward / self.nb_draws + np.sqrt(4 * np.log(np.maximum(1, 2**(self.p + self.i) / (self.nb_arms * self.nb_draws))) / self.nb_draws)
        choice = np.random.choice(np.flatnonzero(index == np.amax(index)))
        self.temp_choice = choice
        
        return self.arms[choice]

    
class MeDZO_IAB:
    """ MeDZO algorithm for infinite-armed bandit problems. """
    
    def __init__(self, horizon, B, c=4):
        self.horizon = horizon
        self.p = ceil(log2(B))
        self.c = c
        
    def start_game(self):
        self.i = 1
        self.t = 1
        self.nb_arms = 2**(self.p + 2 - self.i)
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.arms = np.random.choice(self.horizon, size=self.nb_arms, replace=False)
        self.previous_arms = [self.arms]
        self.probas = []

    def restart(self):
        self.t = 1
        self.probas += [self.nb_draws / np.sum(self.nb_draws)]
        self.nb_arms = 2**(self.p + 2 - self.i)
        self.nb_draws = np.zeros(self.nb_arms + self.i-1)
        self.cum_reward = np.zeros(self.nb_arms + self.i-1)
        self.arms = np.arange(1, self.nb_arms+1) / self.nb_arms
        self.previous_arms += [self.arms]

    def choice(self):
        if self.t <= self.nb_arms + self.i-1:
            self.temp_choice = self.t - 1
        else:
            index = self.cum_reward / self.nb_draws + np.sqrt(self.c * np.log(np.maximum(1, 2**(self.p + self.i) / (self.nb_arms * self.nb_draws))) / self.nb_draws)
            self.temp_choice = np.random.choice(np.flatnonzero(index == np.amax(index)))
        
        if self.temp_choice < self.nb_arms:
            return self.arms[self.temp_choice]
        else:
            j = self.temp_choice - self.nb_arms
            temp_nb_arms = len(self.previous_arms[j]) + j
            temp = np.random.choice(temp_nb_arms, p=self.probas[j])
            if temp < len(self.previous_arms[j]):
                return self.previous_arms[j][temp]
            else:
                for k in range(j-1, 0, -1):
                    temp_nb_arms = len(self.previous_arms[k]) + k
                    temp = np.random.choice(temp_nb_arms, p=self.probas[k])
                    if temp < len(self.previous_arms[j]):
                        return self.previous_arms[k][temp]
                temp = np.random.choice(len(self.previous_arms[0]), p=self.probas[0])
                return self.previous_arms[0][temp]
        
    def get_reward(self, arm, reward):
        self.nb_draws[self.temp_choice] += 1
        self.cum_reward[self.temp_choice] += reward
            
        self.t += 1
        if self.t > 2**(self.p + self.i):
            self.i +=1
            self.restart()
            
            
class empMeDZO_IAB(MeDZO_IAB):
    """ Empirical version of MeDZO for infinite-armed bandit problems. """
    
    def start_game(self):
        self.i = 1
        self.t = 1
        self.nb_arms = 2**(self.p + 2 - self.i)
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.arms = np.random.choice(self.horizon, size=self.nb_arms, replace=False)

    def restart(self):
        self.t = 1
        self.nb_arms = 2**(self.p + 2 - self.i)
        
        index = self.cum_reward / self.nb_draws
        ind = np.argsort(-index)[:self.nb_arms]
        self.arms = self.arms[ind]
        self.cum_reward = self.cum_reward[ind]
        self.nb_draws = self.nb_draws[ind]
    
    def choice(self):
        if self.i == 1 and self.t <= self.nb_arms:
            self.temp_choice = self.t - 1
            return self.arms[self.t - 1]
        
        index = self.cum_reward / self.nb_draws + np.sqrt(self.c * np.log(np.maximum(1, 2**(self.p + self.i) / (self.nb_arms * self.nb_draws))) / self.nb_draws)
        choice = np.random.choice(np.flatnonzero(index == np.amax(index)))
        self.temp_choice = choice
        
        return self.arms[choice]
    
    
class MeDZO_MAB:
    """ MeDZO algorithm for many-armed bandit problems. """
    
    def __init__(self, nb_arms, horizon, beta=0.5, c=4):
        self.true_nb_arms = nb_arms
        self.horizon = horizon
        self.p = ceil(log2(horizon**beta))
        self.c = c
        
    def start_game(self):
        self.i = 1
        self.t = 1
        self.nb_arms = min(2**(self.p + 2 - self.i), self.true_nb_arms)
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.arms = np.random.choice(self.true_nb_arms, size=self.nb_arms, replace=False)
        self.previous_arms = [self.arms]
        self.probas = []

    def restart(self):
        self.t = 1
        self.probas += [self.nb_draws / np.sum(self.nb_draws)]
        self.nb_arms = min(2**(self.p + 2 - self.i), self.true_nb_arms)
        self.nb_draws = np.zeros(self.nb_arms + self.i-1)
        self.cum_reward = np.zeros(self.nb_arms + self.i-1)
        self.arms = np.random.choice(self.true_nb_arms, size=self.nb_arms, replace=False)
        self.previous_arms += [self.arms]

    def choice(self):
        if self.t <= self.nb_arms + self.i-1:
            self.temp_choice = self.t - 1
        else:
            index = self.cum_reward / self.nb_draws + np.sqrt(self.c * np.log(np.maximum(1, 2**(self.p + self.i) / (self.nb_arms * self.nb_draws))) / self.nb_draws)
            self.temp_choice = np.random.choice(np.flatnonzero(index == np.amax(index)))
        
        if self.temp_choice < self.nb_arms:
            return self.arms[self.temp_choice]
        else:
            j = self.temp_choice - self.nb_arms
            temp_nb_arms = len(self.previous_arms[j]) + j
            temp = np.random.choice(temp_nb_arms, p=self.probas[j])
            if temp < len(self.previous_arms[j]):
                return self.previous_arms[j][temp]
            else:
                for k in range(j-1, 0, -1):
                    temp_nb_arms = len(self.previous_arms[k]) + k
                    temp = np.random.choice(temp_nb_arms, p=self.probas[k])
                    if temp < len(self.previous_arms[j]):
                        return self.previous_arms[k][temp]
                temp = np.random.choice(len(self.previous_arms[0]), p=self.probas[0])
                return self.previous_arms[0][temp]
        
    def get_reward(self, arm, reward):
        self.nb_draws[self.temp_choice] += 1
        self.cum_reward[self.temp_choice] += reward
            
        self.t += 1
        if self.t > 2**(self.p + self.i):
            self.i +=1
            self.restart()
            
            
class empMeDZO_MAB(MeDZO_MAB):
    """ Empirical version of MeDZO for many-armed bandit problems. """
    
    def start_game(self):
        self.i = 1
        self.t = 1
        self.nb_arms = min(2**(self.p + 2 - self.i), self.true_nb_arms)
        self.nb_draws = np.zeros(self.nb_arms)
        self.cum_reward = np.zeros(self.nb_arms)
        self.arms = np.random.choice(self.true_nb_arms, size=self.nb_arms, replace=False)

    def restart(self):
        self.t = 1
        self.nb_arms = min(2**(self.p + 2 - self.i), self.true_nb_arms)
        
        index = self.cum_reward / self.nb_draws
        ind = np.argsort(-index)[:self.nb_arms]
        self.arms = self.arms[ind]
        self.cum_reward = self.cum_reward[ind]
        self.nb_draws = self.nb_draws[ind]
    
    def choice(self):
        if self.i == 1 and self.t <= self.nb_arms:
            self.temp_choice = self.t - 1
            return self.arms[self.t - 1]
        
        index = self.cum_reward / self.nb_draws + np.sqrt(self.c * np.log(np.maximum(1, 2**(self.p + self.i) / (self.nb_arms * self.nb_draws))) / self.nb_draws)
        choice = np.random.choice(np.flatnonzero(index == np.amax(index)))
        self.temp_choice = choice
        
        return self.arms[choice]