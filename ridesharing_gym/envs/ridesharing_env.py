import gym
import gym.spaces as spaces
import numpy as np

class RidesharingEnv(gym.Env):

    grid_size = 25
    capacity = 20

    def __init__(self):

        self.action_space = spaces.Discrete(6) # N,S,E,W, center, and reject
        # due to gym limitations must hardcode these parameters
        self.width = 5
        self.length = 5
        self.grid_size = self.width * self.length
        self.capacity = 20

        #init_state = np.zeros(25)
        #init_state[12] = 10
        init_state = 4*np.ones(25)

        # save the initial state for calls to self.reset()
        self.init_state = init_state.astype('int8')

        self.grid_state = init_state.astype('int8')

        self.request_state = self._draw_request()

        self.observation_space = spaces.Tuple((spaces.MultiDiscrete(np.tile(self.capacity, self.grid_size)), 
                                               spaces.MultiDiscrete(np.array([self.grid_size, self.grid_size]))))
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

        loc = self._get_loc(request_start, action)

        # move cars around if needed
        if action == 0: # reject
            pass
        else:
            self._update_car(loc, -1)
            self._update_car(request_end, 1)
            reward = 1.0

        # draw new request
        self.request_state = self._draw_request()

        return ((self.grid_state, self.request_state),
                reward,
                False, # for now, don't worry about episodes
                {})

    def _get_loc(self, origin, action):
        """
        Returna location index given a center and action.
        If exceeds boundary of the grid, returns -1.
        """
        loc = origin

        if action == 2: # N
            loc = origin - self.width
        if action == 3: # S
            loc = origin + self.width
        if action == 4: # E
            loc = origin + 1
        if action == 5: # W
            loc = origin - 1
        if loc < 0 or loc > self.grid_size:
            loc = -1

        return loc

    def _update_car(self, location, change):
        """
        Checks capacity and updates location.
        """
        if location == -1:
            raise Exception('Illegal movement. Location out of bound!')

        update = self.grid_state[location] + change

        if update < 0:
            raise Exception('Illegal movement. Number of cars below zero from location ', location)
        elif update > self.capacity:
            raise Exception('Illegal movement. Number of cars beyond capacity from location ', location)
        else: 
            self.grid_state[location] += update

        return


    def reset(self):
        self.grid_state = self.init_state
        self.request_state = self._draw_request()
        return (self.grid_state, self.request_state)

    def render(self, mode='human'):
        raise NotImplementedError

