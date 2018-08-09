import gym
import gym.spaces as spaces
import numpy as np

class RidesharingEnv(gym.Env):

    def __init__(self, grid_size, capacity):
        self.action_space = spaces.Discrete(6) # N,S,E,W, center, and reject
        self.observation_space = spaces.MultiDiscrete(np.tile(capacity, grid_size))
        # not done

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

