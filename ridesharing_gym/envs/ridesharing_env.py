import gym
import gym.spaces as spaces
import numpy as np
import numpy.random as npr

class RidesharingEnv(gym.Env):

    grid_size = 25
    capacity = 20

    def __init__(self):

        self.action_space = spaces.Discrete(6) # N,S,E,W, center, and reject
        # due to gym limitations must hardcode these parameters
        self.grid_size = 25
        self.capacity = 20

        init_state = np.zeros(25)
        init_state[0] = 10

        # save the initial state for calls to self.reset()
        self.init_state = init_state.astype('int8')

        self.grid_state = init_state.astype('int8')

        self.request_state = self._draw_request()

        self.observation_space = spaces.Tuple((spaces.MultiDiscrete(np.tile(self.capacity, self.grid_size)), spaces.MultiDiscrete(np.array([self.grid_size, self.grid_size]))))
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

        request_start = self.request_state[0]
        request_end = self.request_state[1]

        # move cars around if needed
        if action == 0: # reject
            pass
        if action == 1: # center
            self.grid_state[request_start] -= 1
            self.grid_state[request_end] += 1
            reward = 1.0
        if action == 2: # N
            # note: this is wrong. we aren't dispatching from
            # the center. this should be somewhere other than
            # request_start
            self.grid_state[request_start] -= 1
            self.grid_state[request_end] += 1
            reward = 1.0
        if action == 3: # S
            self.grid_state[request_start] -= 1
            self.grid_state[request_end] += 1
            reward = 1.0
        if action == 4: # E
            self.grid_state[request_start] -= 1
            self.grid_state[request_end] += 1
            reward = 1.0
        if action == 5: # W
            self.grid_state[request_start] -= 1
            self.grid_state[request_end] += 1
            reward = 1.0

        # draw new request
        self.request_state = self._draw_request()

        return ((self.grid_state, self.request_state),
                reward,
                False, # for now, don't worry about episodes
                {})

    def reset(self):
        self.grid_state = self.init_state
        self.request_state = self._draw_request()
        return (self.grid_state, self.request_state)

    def render(self, mode='human'):
        raise NotImplementedError

