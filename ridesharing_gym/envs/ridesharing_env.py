import gym
import gym.spaces as spaces
import numpy as np
from ridesharing_gym.util import GridParameters


class RidesharingEnv(gym.Env):


    def __init__(self):

        self.action_space = spaces.Discrete(6) # N,S,E,W, center, and reject
        # due to gym limitations must hardcode these parameters
        self.grid = GridParameters(2, 2, 2)
        self.euclid = False

        init_state = np.zeros(4)
        init_state[0] = 1

        # save the initial state for calls to self.reset()
        self.init_state = init_state.astype('int8')

        self.grid_state = np.copy(init_state.astype('int8'))

        self.request_state = self._draw_request()

        self.observation_space = spaces.Tuple((spaces.MultiDiscrete(np.tile(self.grid.capacity, self.grid.grid_size)), 
                                               spaces.MultiDiscrete(np.array([self.grid.grid_size, self.grid.grid_size]))))
        # not done

    def _draw_request(self):
        """
        A method to randomly sample a request between two pairs of locations.
        Currently, draws uniformly at random.
        """
        return np.random.randint(self.grid.grid_size, size=2, dtype='int8')


    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.0

        request_start = self.request_state[0]
        request_end = self.request_state[1]

        loc = self.grid.get_loc(request_start, action)

        # move cars around if needed
        if action == 0: # reject
            pass
        elif self._update_car(loc, -1) < 0 or self._update_car(request_end, 1) < 0:
            reward = -9999
        else:
            reward = self._get_reward(loc, request_end)

        # draw new request
        self.request_state = self._draw_request()

        return ((np.copy(self.grid_state), np.copy(self.request_state)),
                reward,
                False, # for now, don't worry about episodes
                {})

    def _update_car(self, location, change):
        """
        Checks capacity and updates location.
        """
        if location == -1:
            return -1

        update = self.grid_state[location] + change

        if update < 0 or update > self.grid.capacity:
            return -1
        else:
            return 1


    def _get_reward(self, start, end, c=1):
        """
        Returns the reward score.
        """
        dist = self.grid.get_dist(start, end)
        reward = dist * c
        if not self.euclid:
            reward = 1.0

        return reward

    def reset(self):
        self.grid_state = np.copy(self.init_state)
        self.request_state = self._draw_request()
        return (np.copy(self.grid_state), np.copy(self.request_state))

    def render(self, mode='human'):
        raise NotImplementedError

