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

    def _grad_theta_alpha(self):
        return self.alpha * (1 - self.alpha)

    def _valid_move(self, source, cars, action):
        # assumes destination has capacity
        act_index = self.grid.get_loc(source, action)
        if (act_index == -1) or (cars[act_index] == 0):
            return False
        else:
            return True

    def _action_probs_array(self, observation):
        probs_array = np.zeros(6)
        cars, request = observation
        source = request[0]
        dest = request[1]

        # if no capacity at destination, must reject
        if cars[dest] >= self.grid.capacity:
            probs_array[0] = 1.0
            return probs_array

        # check if can match center
        if self._valid_move(source, cars, 1):
            probs_array[1] = self.alpha
        else:
            probs_array[0] += self.alpha

        # check if can match neighbors
        neighbor_actions = [2,3,4,5]
        for action in neighbor_actions:
            if self._valid_move(source, cars, action):
                probs_array[action] = (1.0 - self.alpha) / 4.0
            else:
                probs_array[0] += (1.0 - self.alpha) / 4.0

        return probs_array

    def prob_of_action(self, observation, action):
        return self._action_probs_array(observation)[action]

    def _grad_probs_array(self, observation):
        grad_array = np.zeros(6)
        cars, request = observation
        source = request[0]
        dest = request[1]

        moves_valid = np.zeros(6)
        moves_valid[0] = 1
        if not(cars[dest] >= self.grid.capacity):
            for action in range(1,6):
                moves_valid[action] = float(self._valid_move(source, cars, action))

        # center
        grad_array[1] = moves_valid[1] * self._grad_theta_alpha()

        # neighbors
        neighbor_actions = [2,3,4,5]
        for act in neighbor_actions:
            grad_array[act] = moves_valid[act] * (-0.25) * self._grad_theta_alpha()

        # rejection
        grad_array[0] += (1.0 - moves_valid[1]) * self._grad_theta_alpha()
        for act in neighbor_actions:
            grad_array[0] += (1.0 - moves_valid[act]) * (-0.25 * self._grad_theta_alpha())

        return grad_array

    def grad_prob_of_action(self, observation, action):
        return self._grad_probs_array(observation)[action]

    def act(self, observation):
        return np.random.choice([0,1,2,3,4,5], p=self._action_probs_array(observation))
