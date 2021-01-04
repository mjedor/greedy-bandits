#from random import betavariate
from numpy.random import beta, rand
from scipy.special import btdtri


class Beta:
    """ Manipulate posteriors of Bernoulli/Beta experiments. """

    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b
        self.params = [a, b]
        
    def reset(self, a=0, b=0):
        if a == 0:
            a = self.a
        if b == 0:
            b = self.b
        self.params = [a, b]

    def update(self, obs):
        if rand() <= obs:
            temp = 1
        else:
            temp = 0
        self.params[temp] += 1
        
    def sample(self):
        #return betavariate(self.params[1], self.params[0])
        return beta(self.params[1], self.params[0])

    def quantile(self, p):
        return btdtri(self.params[1], self.params[0], p)
