import gym
import ridesharing_gym
import numpy as np


env = gym.make('ridesharing-v0')



def policyEva(env, gamma=0.9, theta=0.01):
	"""
	method for policy evaluation
	"""

    num_states = env._get_num_states()
    num_actions = env.action_space.n
    vf = np.zeros(num_states)
    
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)

def policyIter():
	"""
	method for policy iteration
	"""



