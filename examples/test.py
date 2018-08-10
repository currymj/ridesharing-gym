import gym
import ridesharing_gym

env = gym.make('ridesharing-v0')

observed_state = env.reset()

# for now just take the grid parameters directly out of env
sampling_policy = ridesharing_gym.policies.SamplingPolicy(0.8, env.grid)

for i in range(10):
    print(observed_state)
    act = sampling_policy.act(observed_state)
    print('Randomly chosen action: {}'.format(act))
    observed_state, reward, _, _ = env.step(act)
