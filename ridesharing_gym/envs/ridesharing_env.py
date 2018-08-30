import gym
import gym.spaces as spaces
import numpy as np
from ridesharing_gym.util import GridParameters, read_requests_csv


class RidesharingEnv(gym.Env):


    def __init__(self):

        self.action_space = spaces.Discrete(6) # N,S,E,W, center, and reject
        # due to gym limitations must hardcode these parameters
        self.grid = GridParameters(21, 11, 50)
        self.euclid = False

        init_state = 1.0*np.ones(self.grid.grid_size)

        # save the initial state for calls to self.reset()
        self.init_state = init_state.astype('int8')
        #self.request_probabilities = (1.0 / (self.grid.grid_size * self.grid.grid_size)) * np.ones(self.grid.grid_size * self.grid.grid_size)
        #self.request_array = np.stack([np.array([x, y], dtype='int64')
                                       #for x in range(self.grid.grid_size)
                                       #for y in range(self.grid.grid_size)])

        # bad! bad! don't hardcode this!
        probabilities_dict, _ = read_requests_csv('/Users/curry/src/ridesharing-gym/ridesharing_gym/envs/request_rates.csv', self.grid)

        r_array = list(probabilities_dict.keys())
        self.request_array = np.stack([np.array([x[0], x[1]], dtype='int64') for x in r_array])
        self.request_probabilities = np.array([probabilities_dict[x] for x in r_array])


        print(self.request_array.shape)
        print(self.request_probabilities.shape)
        print(self.grid.grid_size * self.grid.grid_size)

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
        #return np.random.randint(self.grid.grid_size, size=2, dtype='int8')
        chosen_index = np.random.choice(self.request_array.shape[0], p=self.request_probabilities)
        chosen_request = self.request_array[chosen_index,:]
        return chosen_request
    



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
            reward = 0.0
        else:
            self._update_car_mutate(loc, -1)
            self._update_car_mutate(request_end, 1)
            reward = self._get_reward(loc, request_end)

        # draw new request
        self.request_state = self._draw_request()

        return ((np.copy(self.grid_state), np.copy(self.request_state)),
                reward,
                False, # for now, don't worry about episodes
                {})

    def _update_car(self, location, change):
        """
        Checks capacity 
        """
        if location == -1:
            return -1

        update = self.grid_state[location] + change

        if update < 0 or update > self.grid.capacity:
            return -1
        else:
            return 1

    def _update_car_mutate(self, location, change):
        if location == -1:
            raise Exception("shouldn't be updating!")

        update = self.grid_state[location] + change

        if update < 0 or update > self.grid.capacity:
            raise Exception("shouldn't be updating!")
        else:
            self.grid_state[location] += change


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

