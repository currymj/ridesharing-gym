import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from tabular import *

env = gym.make('ridesharing-v0')

vf = valueIteration(env)
opt_policy = get_policy(env, vf)

observed_state = env.reset()
observed_state_index = env.b_map[(tuple(observed_state[0]), tuple(observed_state[1]))]
loc_vec, end_vec = [], []

for i in range(30):
    action = int(opt_policy[observed_state_index])
    _, _, _, _, observed_state_index, loc, request_end = env.step(action, detail=True)
    loc_vec.append(loc)
    end_vec.append(request_end)

def vec_to_fig(input_vector, fig_name):
	counts = np.array([input_vector.count(x) for x in range(4)]).reshape(2, 2)
	map = sns.heatmap(counts)
	fig = map.get_figure()
	fig.savefig("examples/figures/" + fig_name + ".png")
	fig.clf()

vec_to_fig(end_vec, "ending_map")
vec_to_fig(loc_vec, "starting_map")
