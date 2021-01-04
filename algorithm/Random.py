from random import randrange


class Random:
    """ Random policy
    """
    def __init__(self, nb_arms):
        self.nb_arms = nb_arms

    def start_game(self):
        pass

    def choice(self):
        return randrange(self.nb_arms)

    def get_reward(self, arm, reward):
        pass
