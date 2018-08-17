import gym
import ridesharing_gym
from ridesharing_gym.util import state_to_index
import numpy as np


env = gym.make('ridesharing-v0')


def tabular(env, gamma=0.999):
    """
    Implementing value iteration
    """

	num_states = env.observation_space.n
	num_actions = env.action_space.n
    vf = np.zeros(num_states) #value function
    loop = 1000
    epsilon = 0.001
    diff = -1.0
    
    for i in range(lopp):
        vf_p = np.copy(vf) #previous value function
        for s in range(num_states):
            Q_sa = [] #init Q-value for state action pair
            for a in range(num_actions): 
                next_rewards = []
                for next_tuple in env.P[s][a]: 
                    prob, next_state, reward = next_tuple 
                    next_rewards.append((prob * (reward + vf_p[next_state])))
                Q_sa.append(np.sum(next_rewards))
            vf[s] = max(Q_sa)
            
        if(np.abs(np.abs(np.sum(vf_p - vf)) - diff) < epsilon):
            print('Converges at iteration %d' % (i+1))
            break
        diff = np.abs(np.sum(vf_p - vf))
    return vf