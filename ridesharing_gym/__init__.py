from gym.envs.registration import register

register(
    id='ridesharing-v0',
    entry_point='ridesharing_gym.envs:RidesharingEnv'
)
