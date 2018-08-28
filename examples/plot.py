import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from tabular import *

env = gym.make('ridesharing-v0')

vf = valueIteration(env)
opt_policy = get_policy(env, vf)

observed_state = env.reset()
observed_state_index = env.b_map[(tuple(observed_state[0]), tuple(observed_state[1]))]
loc_vec = []

for i in range(30):
    action = int(opt_policy[observed_state_index])
    _, _, _, _, observed_state_index, loc = env.step(action, detail=True)
    loc_vec.append(loc)


counts = [loc_vec.count(x) for x in range(4)]
counts = np.array(counts)
count_matrix = counts.reshape(2, 2)
starting_loc_map = sns.heatmap(count_matrix)
starting_loc_fig = starting_loc_map.get_figure()
starting_loc_fig.savefig("starting_map.png")