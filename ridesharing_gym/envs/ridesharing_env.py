import gym
import gym.spaces as spaces
import numpy as np
from ridesharing_gym.util import GridParameters


class RidesharingEnv(gym.Env):


    def __init__(self):

        self.action_space = spaces.Discrete(6) # N,S,E,W, center, and reject
        # due to gym limitations must hardcode these parameters
        self.grid = GridParameters(5, 5, 20)
        self.euclid = False

        init_state = np.zeros(self.grid.grid_size)
        init_state[12] = 10
        #init_state = 4*np.ones(25)

        # save the initial state for calls to self.reset()
        self.init_state = init_state.astype('int8')

        self.grid_state = np.copy(init_state.astype('int8'))

        self.request_state = self._draw_request()

        self.observation_space = spaces.Tuple((spaces.MultiDiscrete(np.tile(self.grid.capacity, self.grid.grid_size)), 
                                               spaces.MultiDiscrete(np.array([self.grid.grid_size, self.grid.grid_size]))))
        self.P = self._get_P()
        
        self.f_map, self.b_map = self._get_maps()


    def _get_P(self):
        """
        Returns a transition function for a given state action pair
        Return for each pair the probability, next state and reward
        """

        num_state = self.observation_space.n
        num_action = self.action_space.n

        #initialize the P matrix
        P = np.zeros((num_state, num_action), dtype=object)

        for s in range(num_state):
            for a in range(num_action):
                P[s][a] = 

        return P


    def _get_maps(self):
        """
        Iterate over all states
        Returns forward and backward mapping of state and index
        """
        c = self.grid.capacity
        s = self.grid.grid_size

        # matrix of all states
        states = [(i, x) for i in np.ndindex(tuple(np.tile(c, s))) for x in np.ndindex(s, s)]
        
        #forward and backward mapping of state and index
        f_map = dict(enumerate(states))
        b_map = {v: k for k, v in f_map.items()}

        return f_map, b_map


    def _draw_request(self):
        """
        A method to randomly sample a request between two pairs of locations.
        Currently, draws uniformly at random.
        """
        return np.random.randint(self.gird.grid_size, size=2, dtype='int8')


    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.0

        request_start = self.request_state[0]
        request_end = self.request_state[1]

        loc = self.grid.get_loc(request_start, action)

        # move cars around if needed
        if action == 0: # reject
            pass
        else:
            self._update_car(loc, -1)
            self._update_car(request_end, 1)
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
            raise Exception('Illegal movement. Location out of bound!')

        update = self.grid_state[location] + change

        if update < 0:
            raise Exception('Illegal movement. Number of cars below zero from location ', location)
        elif update > self.grid.capacity:
            raise Exception('Illegal movement. Number of cars beyond capacity from location ', location)
        else:
            self.grid_state[location] += change

        return

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

