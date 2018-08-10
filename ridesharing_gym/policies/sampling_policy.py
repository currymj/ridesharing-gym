import numpy as np

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
