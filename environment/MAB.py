import numpy as np


class Result:
    """ The Result class for analyzing the output of bandit experiments. """

    def __init__(self, nb_arms, horizon):
        self.nb_arms = nb_arms
        self.choices = np.zeros(horizon, dtype=np.int)
        self.rewards = np.zeros(horizon)

    def store(self, t, choice, reward):
        self.choices[t] = choice
        self.rewards[t] = reward

    def get_nb_pulls(self):
        nb_pulls = np.zeros(self.nb_arms)
        
        for choice in self.choices:
            nb_pulls[choice] += 1
            
        return nb_pulls


class MAB:
    """ Multi-armed bandit problem with arms given in the 'arms' list """
    
    def __init__(self, arms):
        self.arms = arms
        self.nb_arms = len(arms)

    def play(self, algorithm, horizon):
        algorithm.start_game()
        result = Result(self.nb_arms, horizon)
        
        for t in range(horizon):
            choice = algorithm.choice()
            reward = self.arms[choice].draw()
            algorithm.get_reward(choice, reward)
            result.store(t, choice, self.arms[choice].expectation)
        
        return result


class EvaluationMAB:
    """ Evaluation class for a multi-armed bandit problem """
    
    def __init__(self, env, algorithm, nb_repetitions, horizon, tsav=[]):
        if len(tsav) > 0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(horizon)
            
        self.env = env
        self.nb_repetitions = nb_repetitions
        self.cum_reward = np.zeros((self.nb_repetitions, len(self.tsav)))

        for k in range(nb_repetitions):
            if nb_repetitions < 10 or k % (nb_repetitions / 10) == 0:
                print(k)
            
            result = env.play(algorithm, horizon)
            self.cum_reward[k, :] = np.cumsum(result.rewards)[self.tsav]

    def mean_reward(self):
        return sum(self.cum_reward[:, -1]) / self.nb_repetitions

    def std_regret(self):
        temp = (1 + self.tsav) * max([arm.expectation for arm in self.env.arms])
        return np.std(temp - self.cum_reward, 0)

    def mean_regret(self):
        return (1 + self.tsav) * max([arm.expectation for arm in self.env.arms]) - np.mean(self.cum_reward, 0)
