import gym
import gym.spaces as spaces
import numpy as np
import numpy.random as npr

class RidesharingEnv(gym.Env):

    def __init__(self, grid_size, capacity, init_state):

        self.action_space = spaces.Discrete(6) # N,S,E,W, center, and reject
        self.grid_size = grid_size
        self.capacity = capacity


        # save the initial state for resetting
        self.init_state = init_state.astype('int8')

        self.grid_state = init_state.astype('int8')
        self.request_state = self._draw_request()

        self.observation_space = spaces.Tuple((spaces.MultiDiscrete(np.tile(capacity, grid_size)), spaces.MultiDiscrete(np.array([grid_size, grid_size]))))
        # not done

    def _draw_request(self):
        """
        A method to randomly sample a request between two pairs of locations.
        Currently, draws uniformly at random.
        """
        return np.random.randint(25, size=2, dtype='int8')


    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.0
        raise NotImplementedError

        # move cars around if needed
        if action == 0: # reject
            pass
        if action == 1: # center
            pass
        if action == 2: # N
            pass
        if action == 3: # S
            pass
        if action == 4: # E
            pass
        if action == 5: # W
            pass

        # draw new request
        self.request_state = self._draw_request()

        return ((self.grid_state, self.request_state),
                reward,
                False, # for now, don't worry about episodes
                {})

    def reset(self):
        self.grid_state = self.init_state
        self.request_state = self._draw_request()

    def render(self, mode='human'):
        raise NotImplementedError

