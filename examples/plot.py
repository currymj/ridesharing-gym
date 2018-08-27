import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from tabular import *

env = gym.make('ridesharing-v0')

vf = valueIteration(env)
opt_policy = get_policy(env, vf)

observed_state = env.reset();
observed_state_index = env.b_map[(tuple(observed_state[0]), tuple(observed_state[1]))]

for i in range(10):
	action = opt_policy[observed_state_index]
    _, _, _, _, observed_state_index, loc = env.step(action, detail=True)
    loc_vec[i] = loc


counts = [loc_vector.count(x) for x in range(4)]
counts = np.array(counts)
count_matrix = counts.reshape(2, 2)
ax = sns.heatmap(count_matrix)