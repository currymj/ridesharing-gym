import gym
import ridesharing_gym
import numpy as np


env = gym.make('ridesharing-v0')



def policyEva(env, gamma=0.9, theta=0.01, diff=100):
	"""
	method for policy evaluation
	"""

    num_states = env._get_num_states()
    num_actions = env.action_space.n
    vf = np.zeros(num_states)


    for s in range(num_states):
        v = 0
        # Look at the possible next actions
        for a, a_prob in enumerate(policy[s]):
            # For each action, look at the possible next states...
            for  prob, next_state, reward, _ in env.P[s][a]:
                # Calculate the expected value
                v += a_prob * prob * (reward + gamma * vf[next_state])
        # How much our value function changed (across any states)
        diff = max(diff, np.abs(v - vf[s]))
        vf[s] = v
    # Stop evaluating once our value function change is below a threshold
    if diff < theta:
        break
    return np.array(vf)

def policyIter():
	"""
	method for policy iteration
	"""



