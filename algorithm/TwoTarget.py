import numpy as np
from math import floor


class TwoTarget:
    """
        Ref: Two-target algorithms for infinite-armed bandits with bernoulli rewards. Bonald, T., & Proutiere, A. (2013).
    """
    
    def __init__(self, horizon, m, alpha=1, beta=1):
        self.horizon = horizon
        self.m = m
        self.l_1 = floor((alpha * horizon / (beta + 1))**(1 /(beta + 2)))
        self.l_2 = floor(m * (alpha * horizon / (beta + 1))**(1 / (beta + 1)))
        
    def start_game(self):
        self.index = np.random.choice(self.horizon, size=self.horizon, replace=False)
        self.I = 0
        self.L = 0
        self.M = 0
        self.exploit = False
    
    def explore(self):
        self.I += 1
        self.L = 0
        self.M = 0
        
    def choice(self):
        return self.index[self.I]
    
    def get_reward(self, arm, reward):
        if not self.exploit:
            if reward == 1:
                self.L += 1
            else:
                self.M +=1
                if self.M == 1:
                    if self.L < self.l_1:
                        self.explore()
                elif self.M == self.m:
                    if self.L < self.l_2:
                        self.explore()
                    else:
                        self.exploit = True
