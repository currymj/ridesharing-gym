import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from tabular import *

env = gym.make('ridesharing-v0')

vf = valueIteration(env)
opt_policy = get_policy(env, vf)

observed_state = env.reset()
observed_state_index = env.b_map[(tuple(observed_state[0]), tuple(observed_state[1]))]
loc_vec, end_vec, occupied_vec = [], [], []

num_period = 500


for i in range(num_period):
    grid_state = observed_state[0]
    for j in range(4):
        if grid_state[j] > 0:
            occupied_vec.append(j)

    action = int(opt_policy[observed_state_index])
    observed_state, _, _, _, observed_state_index, loc, request_end = env.step(action, detail=True)
    
    if action != 0:
        loc_vec.append(loc)
        end_vec.append(request_end)


volume_vec = loc_vec + end_vec


def vec_to_fig(input_vector, fig_name, num_period):
    counts = np.array([input_vector.count(x) for x in range(4)]).reshape(2, 2)
    print(counts)
    rate = np.divide(counts, num_period)
    map = sns.heatmap(rate)
    fig = map.get_figure()
    fig.savefig("examples/figures/" + fig_name + ".png")
    fig.clf()


vec_to_fig(end_vec, "ending_map", 500)
vec_to_fig(loc_vec, "starting_map", 500)
vec_to_fig(volume_vec, "drop_rate", 500)
vec_to_fig(occupied_vec, "time", 500)
