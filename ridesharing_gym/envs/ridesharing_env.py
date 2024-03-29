import gym
import gym.spaces as spaces
import numpy as np
from joblib import Parallel, delayed
from ridesharing_gym.util import GridParameters, read_requests_csv
import math

# from dynamic-ride-sharing repo

def random_ints_with_sum(n, m, c):
    "Generates a length n integer array, all at most  c, of total m."
    samples=np.random.uniform(0, c, n)
    if sum(samples) < m:
        return random_ints_with_sum(n, m, c)
    samples=(m/float(sum(samples)))*samples

    i=0
    j=1
    while True:
        if samples[i].is_integer():
            i=j
            j=j+1
        if j>=n:
            break
        if samples[j].is_integer():
            j=j+1

        if j>=n:
            break

        alpha=min(math.ceil(samples[i])-samples[i], samples[j]-math.floor(samples[j]))
        beta=min(samples[i]-math.floor(samples[i]), math.ceil(samples[j])-samples[j])

        r=np.random.uniform(0, 1)
        if r<=beta/(alpha + beta):
            samples[i]+=alpha
            samples[j]-=alpha
        else:
            samples[i]-=beta
            samples[j]+=beta

    return samples

class RidesharingEnv(gym.Env):

    def __init__(self):

        self.action_space = spaces.Discrete(6) # N,S,E,W, center, and reject
        # due to gym limitations must hardcode these parameters
        self.euclid = False

        init_state = np.zeros(self.grid.grid_size)
        init_state[0] = 1
        self.grid = GridParameters(21, 11, 50)
        self.euclid = False

        init_state = random_ints_with_sum(self.grid.grid_size, 5000, self.grid.capacity)

        # save the initial state for calls to self.reset()
        self.init_state = init_state.astype('int8')

        self.observation_space = spaces.Tuple((spaces.MultiDiscrete(np.tile(self.grid.capacity, self.grid.grid_size)), 
                                               spaces.MultiDiscrete(np.array([self.grid.grid_size, self.grid.grid_size]))))

        self.f_map, self.b_map = self._get_maps()

        self._get_P()
        
        probabilities_dict, weights_dict = read_requests_csv('./ridesharing_gym/envs/request_rates.csv', self.grid)

        self.weights_dict = weights_dict

        r_array = list(probabilities_dict.keys())
        self.request_array = np.stack([np.array([x[0], x[1]], dtype='int64') for x in r_array])
        self.request_probabilities = np.array([probabilities_dict[x] for x in r_array])


        self.grid_state = np.copy(init_state.astype('int8'))

        self.request_state = self._draw_request()



    def _get_P(self):
        """
        Returns a transition function for a given state action pair
        Return for each pair a list of (probability, next state and reward) tuple
        """

        num_states = self._get_num_states()
        num_actions = self.action_space.n
        grid_size = self.grid.grid_size

        #initialize the transition matrix
        self.P = np.zeros((num_states, num_actions), dtype=object)

        #loop over actions
        for a in range(num_actions):
            self._single_a_P(a, num_states, grid_size)
        #Parallel(n_jobs=6)(delayed(self._single_a_P)(a, num_states, grid_size) for a in range(num_actions))   


    def _single_a_P(self, a, num_states, grid_size):
        
        prob = (1.0/grid_size)**2 
        for s in range(num_states):
            #print(s / num_states * 100, a)
            self.P[s][a] = []
            if self._legal_moves(s, a):
                #loop over requests
                for r in np.ndindex((grid_size, grid_size)):
                    next_state, reward = self._step_index(s, a, r)
                    self.P[s][a].append((prob, next_state, reward))


    def _get_num_states(self):
        """
        Computes the number of states
        """
        g = self.grid.grid_size
        c = self.grid.capacity + 1
        num_states = (c**g)*g*g
        return num_states


    def _step_index(self, index, action, request):
        """
        Given state index, action and request
        Returns a new state index
        """
        self.grid_state = np.asarray(self.f_map[index][0])
        self.request_state = np.asarray(self.f_map[index][1])
        next_state, reward, _, _ = self.step(action)
        next_state = (tuple(next_state[0]), tuple(request))
        next_index = self.b_map[next_state]

        return next_index, reward


    def _get_maps(self):
        """
        Iterate over all states
        Returns forward and backward mapping of state and index
        """
        c = self.grid.capacity + 1
        s = self.grid.grid_size

        # matrix of all states
        states = [(i, x) for i in np.ndindex(tuple(np.tile(c, s))) for x in np.ndindex(s, s)]
        
        #forward and backward mapping of state and index
        f_map = dict(enumerate(states))
        b_map = {v: k for k, v in f_map.items()}

        return f_map, b_map


    def _legal_moves(self, state_index, action):
        """
        Returns False if illegal moves
        """
        self.grid_state = np.asarray(self.f_map[state_index][0])
        self.request_state = np.asarray(self.f_map[state_index][1])
        try:    
            self.step(action)
            return True
        except:
            return False


    def _draw_request(self, f='Unif'):
        """
        A method to randomly sample a request between two pairs of locations.
        Default is to draw uniformly at random.
        """

        chosen_index = np.random.choice(self.request_array.shape[0], p=self.request_probabilities)
        chosen_request = self.request_array[chosen_index,:]
        return chosen_request


    def step(self, action, detail=False):
        assert self.action_space.contains(action)
        reward = 0.0

        request_start = self.request_state[0]
        request_end = self.request_state [1]

        loc = self.grid.get_loc(request_start, action)
        # move cars around if needed
        if action == 0: # reject
            pass
        elif self._update_car(loc, -1) < 0 or self._update_car(request_end, 1) < 0:
            reward = 0.0
        else:
            self._update_car_mutate(loc, -1)
            self._update_car_mutate(request_end, 1)
            reward = self._get_reward(request_start, request_end)

        # draw new request
        self.request_state = self._draw_request()

        next_state = (tuple(self.grid_state), tuple(self.request_state))
        observed_state_index = self.b_map[next_state]

        if detail: return ((np.copy(self.grid_state), 
                            np.copy(self.request_state)),
                            reward,
                            False, # for now, don't worry about episodes
                            {},
                            observed_state_index,
                            loc,
                            request_end)
        else: return ((np.copy(self.grid_state), 
                       np.copy(self.request_state)),
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
        reward = self.weights_dict[(start, end)] * c
        if not self.euclid:
            reward = 1.0

        return reward


    def reset(self):
        self.grid_state = np.copy(self.init_state)
        self.request_state = self._draw_request()
        return (np.copy(self.grid_state), np.copy(self.request_state))


    def render(self, mode='human'):
        raise NotImplementedError

