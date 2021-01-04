from scipy.stats import truncexpon
from random import expovariate


class Exponential:
    """ Exponentially distributed arm, truncated """

    def __init__(self, lambd, trunc=1):
        self.lambd = lambd
        self.trunc = trunc
        
        if trunc:
            self.rv = truncexpon(trunc, scale=1/lambd)
            self.expectation = self.rv.mean()
        else:
            self.expectation = 1. / lambd

    def draw(self):
        if self.trunc:
            return self.rv.rvs()
        else:
            return expovariate(self.lambd)
