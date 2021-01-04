from random import paretovariate


class Pareto:
    """ Pareto distributed arm """

    def __init__(self, alpha):
        self.alpha = alpha
        if alpha <= 1:
            self.expectation = float('inf')
        else:
            self.expectation = alpha / (alpha - 1)

    def draw(self):
        return paretovariate(self.alpha)
