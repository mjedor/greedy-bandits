from random import betavariate


class Beta:
    """ Beta distributed arm """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.expectation = a / (a + b)

    def draw(self):
        return betavariate(self.a, self.b)
