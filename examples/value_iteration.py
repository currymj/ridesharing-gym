import gym
import ridesharing_gym
import numpy as np


env = gym.make('ridesharing-v0')


def valueIteration(env, gamma=0.9, loop=10000, epsilon=0.1, diff=100):
    """
    Implementing value iteration
    Returns a value function
    """
    num_states = env._get_num_states()
    num_actions = env.action_space.n
    vf = np.zeros(num_states) #value function
    
    
    for i in range(loop):
        vf_p = np.copy(vf) #previous value function
        for s in range(num_states):
            Q_sa = [] #init Q-value for state action pair
            for a in range(num_actions): 
                next_rewards = []
                for next_tuple in env.P[s][a]: 
                    prob, next_state, reward = next_tuple 
                    next_rewards.append((prob * (reward + gamma*vf_p[next_state])))
                Q_sa.append(np.sum(next_rewards))
            vf[s] = max(Q_sa)
        if(diff < epsilon): 

            print('Converges at iteration %d' % (i+1))
            break
        diff = np.abs(np.sum(vf_p - vf))
    return vf


def get_policy(env, vf, gamma=0.999):
    '''
    Returns policy given the value function
    '''
    num_states = env._get_num_states()
    num_actions = env.action_space.n
    policy = np.zeros(num_states) 

    for s in range(num_states):
        Q_sa = np.zeros(num_actions)
        for a in range(num_actions):
            for next_tuple in env.P[s][a]:
                prob, next_state, reward = next_tuple
                Q_sa[a] += (prob * (reward + gamma * vf[next_state]))
        policy[s] = np.argmax(Q_sa)
    
    retern policy


vf = valueIteration(env)
opt_policy = get_policy(env, vf)
print(opt_policy)






