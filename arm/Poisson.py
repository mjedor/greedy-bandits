from scipy.stats import poisson
from math import isinf, exp


class Poisson:
    """ Poisson distributed arm, possibly truncated """

    def __init__(self, p, trunc=float('inf')):
        self.p = p
        self.trunc = trunc
        if isinf(trunc):
            self.expectation = p
        else:
            q = exp(-p)
            sq = q
            self.expectation = 0
            for k in range(1, self.trunc):
                q = q * p / k
                self.expectation += k * q
                sq += q
            self.expectation += self.trunc * (1 -s q)

    def draw(self):
        return min(poisson.rvs(self.p), self.trunc)
