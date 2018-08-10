from gym.envs.registration import register
import ridesharing_gym.policies

register(
    id='ridesharing-v0',
    entry_point='ridesharing_gym.envs:RidesharingEnv'
)
