import gym
import gym.spaces as spaces
import numpy as np
from ridesharing_gym.util import GridParameters
from joblib import Parallel, delayed

class RidesharingEnv(gym.Env):

    def __init__(self):

        self.action_space = spaces.Discrete(6) # N,S,E,W, center, and reject
        # due to gym limitations must hardcode these parameters
        self.grid = GridParameters(2, 2, 3)
        self.euclid = False

        init_state = np.zeros(self.grid.grid_size)
        init_state[0] = 1

        # save the initial state for calls to self.reset()
        self.init_state = init_state.astype('int8')


        self.observation_space = spaces.Tuple((spaces.MultiDiscrete(np.tile(self.grid.capacity, self.grid.grid_size)), 
                                               spaces.MultiDiscrete(np.array([self.grid.grid_size, self.grid.grid_size]))))

        self.f_map, self.b_map = self._get_maps()

        self._get_P()
        
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
        #if f == 'Pois':
        #not done

        return np.random.randint(self.grid.grid_size, size=2, dtype='int8')


    def step(self, action, detail=False):
        assert self.action_space.contains(action)
        reward = 0.0

        request_start = self.request_state[0]
        request_end = self.request_state [1]

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

