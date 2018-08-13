import numpy as np
import scipy.special

class SamplingPolicy:

    def __init__(self, alpha, grid):
        self.alpha = alpha
        self.grid = grid

    def act(self, observation):
        cars, request = observation
        source = request[0]
        dest = request[1]

        if cars[dest] >= self.grid.capacity:
            return 0

        action_choice = 1
        if np.random.rand() > self.alpha:
            action_choice = np.random.randint(4) + 2

        chosen_index = self.grid.get_loc(source, action_choice)

        if chosen_index == -1:
            return 0 # reject

        if cars[chosen_index] == 0:
            return 0

        return action_choice


# by rights we should probably do inheritance here
# but it's just not worth the trouble
class LogisticSamplingPolicy:
    def __init__(self, theta, grid):
        self.theta = theta
        self.grid = grid

    @property
    def alpha(self):
        return scipy.special.expit(self.theta)

    def prob_of_action(self, observation, action):
        raise NotImplementedError

    def grad_prob_of_action(self, observation, action):
        raise NotImplementedError

    def act(self, observation):
        cars, request = observation
        source = request[0]
        dest = request[1]

        if cars[dest] >= self.grid.capacity:
            return 0

        action_choice = 1
        if np.random.rand() > self.alpha:
            action_choice = np.random.randint(4) + 2

        chosen_index = self.grid.get_loc(source, action_choice)

        if chosen_index == -1:
            return 0 # reject

        if cars[chosen_index] == 0:
            return 0

        return action_choice
