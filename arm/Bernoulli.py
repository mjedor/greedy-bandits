from random import random


class Bernoulli:
    """ Bernoulli distributed arm """

    def __init__(self, p):
        self.p = p
        self.expectation = p
        
    def draw(self):
        return float(random() < self.p)
