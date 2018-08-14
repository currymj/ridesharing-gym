import gym
import ridesharing_gym
import numpy as np


env = gym.make('ridesharing-v0')

observed_state = env.reset()

# for now just take the grid parameters directly out of env
sampling_policy = ridesharing_gym.policies.LogisticSamplingPolicy(0.0, env.grid)
print('Initial alpha: {}'.format(sampling_policy.alpha))

N_EPISODES = 1
T_STEPS = 100
for episode in range(N_EPISODES):
    acts = []
    states = []
    rewards = np.zeros(T_STEPS)
    for t in range(T_STEPS):
        states.append(observed_state)
        act = sampling_policy.act(observed_state)
        acts.append(act)
        print('Randomly chosen action: {}'.format(act))
        print('Prob of chosen action: {}'.format(sampling_policy.prob_of_action(observed_state, act)))
        print('Gradient of prob of chosen action: {}'.format(sampling_policy.grad_prob_of_action(observed_state, act)))
        observed_state, reward, _, _ = env.step(act)
        rewards[t] = reward
    print(acts)
    print(rewards)
