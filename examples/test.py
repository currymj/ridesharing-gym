import gym
import ridesharing_gym

env = gym.make('ridesharing-v0')

observed_state = env.reset()
for i in range(10):
    print(observed_state)
    act = env.action_space.sample()
    print("Randomly chosen action: %d".format(act))
    observed_state, reward, _, _ = env.step(act)
