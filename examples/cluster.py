import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from tabular import *
import matplotlib.pyplot as plt

env = gym.make('ridesharing-v0')

P = env.P
P.dump('examples/p_matrix.temp')
