from scipy.stats import truncnorm
from random import gauss
from math import sqrt


class Gaussian:
    """ Gaussian distributed arm """

    def __init__(self, mu, sigma2=1, trunc=False, lower=0., upper=1.):
        self.sigma2 = sigma2
        self.mu = mu
        self.trunc = trunc
        self.lower = lower
        self.upper = upper

        sigma = sqrt(sigma2)

        if trunc is True:
            a = (lower - mu) / sigma
            b = (upper - mu) / sigma
            self.rv = truncnorm(a, b, loc=mu, scale=sigma)
            self.expectation = self.rv.mean()
        else:
            self.expectation = mu
        
    def draw(self):
        if self.trunc is True:
            return self.rv.rvs()
        else:
            return gauss(self.mu, sqrt(self.sigma2))
