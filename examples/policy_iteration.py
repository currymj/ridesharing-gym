import gym
import ridesharing_gym
import numpy as np
from value_iteration import *

env = gym.make('ridesharing-v0')



def policyEva(policy_prob, env, gamma=0.9, theta=0.01, diff=100):
    """
    method for policy evaluation
    """

    num_states = env._get_num_states()
    num_actions = env.action_space.n
    vf = np.zeros(num_states)

    while True:
        diff = 0
        for s in range(num_states):
            v = 0
            for a, a_prob in enumerate(policy_prob[s]):
                for  prob, next_state, reward in env.P[s][a]:
                    v += a_prob * prob * (reward + gamma * vf[next_state])
            diff = max(diff, np.abs(v - vf[s]))
            vf[s] = v
        if diff < theta:
            break
    return np.array(vf)


def policy_to_mat(policy):
    """
    method to convert a deterministic policy
    return a numpy matrix 
    """
    num_states = env._get_num_states()
    num_actions = env.action_space.n

    policy_prob = np.zeros((num_states, num_actions))

    for s in range(num_states):
        a = int(policy[s])
        policy_prob[s][a] = 1

    return policy_prob


def policyIter(env, gamma=0.9):
    """
    method for policy iteration
    """
    
    num_states = env._get_num_states()
    num_actions = env.action_space.n

    policy_prob = np.zeros((num_states, num_actions))

    while True:

        vf = policyEva(policy_prob, env)
        
        policy_stable = True
    
        for s in range(num_states):
            cur_a = np.argmax(policy_prob[s])
            value = np.zeros(num_actions)
                
            for a in range(num_actions):
                for prob, next_state, reward in env.P[s][a]:
                    value[a] += prob * (reward + gamma * vf[next_state])
            best_a = np.argmax(value)

            if cur_a != best_a:
                policy_stable = False

            policy_prob[s] = np.eye(num_actions)[best_a]

        if policy_stable: return policy_prob, vf 


# vf = valueIteration(env)
# opt_policy = get_policy(env, vf)
# policy_prob = policy_to_mat(opt_policy)
# policy_evaluation = policyEva(policy_prob, env)
# print(policy_evaluation)


policy_prob, vf = policyIter(env)
print(policy_prob)