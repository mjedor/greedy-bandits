import numpy as np


class Zooming:
    def __init__(self, horizon, L=1, alpha=1):
        self.horizon = horizon
        self.L = L
        self.alpha = alpha
        
    def start_game(self):
        self.t = 1
        self.nb_draws = []
        self.cum_reward = []
        self.arms = []
        
    def choice(self):
        ### Activation rule
        if self.t == 1:
            temp =  np.random.rand()
            self.arms += [temp]
            self.nb_draws += [0]
            self.cum_reward += [0]
            return temp
        
        # Compute means and confidence radius
        temp_cum_reward = np.array(self.cum_reward)
        temp_nb_draws = np.array(self.nb_draws)
        
        means = temp_cum_reward / temp_nb_draws
        radius = np.sqrt(2 * np.log(self.horizon) / (temp_nb_draws + 1))
        
        # Activation rule
        a = np.linspace(0, 1, 101, endpoint=True)
        np.random.shuffle(a)
        for elt in a:
            activation = True
            for arm in range(len(self.arms)):
                if self.arms[arm] - (radius[arm]/self.L)**(1/self.alpha) <= elt <= self.arms[arm] + (radius[arm]/self.L)**(1/self.alpha):
                    activation = False
                    break
                    
            if activation:
                self.arms += [elt]
                self.nb_draws += [0]
                self.cum_reward += [0]
                return elt
                
        
        # Selection rule
        index = means + 2 * radius
        ind = np.random.choice(np.flatnonzero(index == np.amax(index)))
        return self.arms[ind]
        
    def get_reward(self, arm, reward):
        index_arm = np.random.choice(np.flatnonzero(np.array(self.arms) == arm))
        self.nb_draws[index_arm] += 1
        self.cum_reward[index_arm] += reward
        self.t += 1
