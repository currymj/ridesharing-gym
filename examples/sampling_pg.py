import gym
import ridesharing_gym
from ridesharing_gym.util import discounted_episode_returns
import numpy as np


def reinforce_episodic_update(sampling_policy, acts, states, rewards, lr=0.01):
    returns = discounted_episode_returns(rewards)
    for t in range(len(acts)):
        #print('t: {}, alpha: {}, theta:{}'.format(t, sampling_policy.alpha, sampling_policy.theta))
        #print('State: {}', states[t])
        #print('Action taken: {}', acts[t])
        grad_val = sampling_policy.grad_prob_of_action(states[t], acts[t])
        prob_val = sampling_policy.prob_of_action(states[t], acts[t])
        update_term = lr * (returns[t] *
                                       (grad_val / prob_val))
        sampling_policy.theta += update_term
        #print('update: {}, Gradient: {}, Probability: {}'.format(update_term, grad_val, prob_val))



env = gym.make('ridesharing-v0')

observed_state = env.reset()

# for now just take the grid parameters directly out of env
sampling_policy = ridesharing_gym.policies.LogisticSamplingPolicy(0.0, env.grid)
print('Initial alpha: {}'.format(sampling_policy.alpha))

N_EPISODES = 100
T_STEPS = 1000
for episode in range(N_EPISODES):
    acts = []
    states = []
    rewards = np.zeros(T_STEPS)
    for t in range(T_STEPS):
        states.append(observed_state)
        act = sampling_policy.act(observed_state)
        acts.append(act)
        #print('Randomly chosen action: {}'.format(act))
        prob_act = sampling_policy.prob_of_action(observed_state, act)
        #print('Prob of chosen action: {}'.format(prob_act))
        assert prob_act > 0.0
        #print('Gradient of prob of chosen action: {}'.format(sampling_policy.grad_prob_of_action(observed_state, act)))
        observed_state, reward, _, _ = env.step(act)
        rewards[t] = reward
    reinforce_episodic_update(sampling_policy, acts, states, rewards)
    print('alpha: {}, theta: {}'.format(sampling_policy.alpha,
                                        sampling_policy.theta))
