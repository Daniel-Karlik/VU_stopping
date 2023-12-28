# This is main script for running examples related with Daniel Karlik's Master Thesis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Environment import Environment, BaseAgent
#from experiment import Experiment
#from FPD_Stop_PE import FPD_Stop_PE
from Learning_FPD_Stop_PE import Experiment

# Defining the task
num_states = 5
num_actions = 7
num_steps = 10
num_mc_runs = 5
seed = 10

# function experiment
#experiment1 = Experiment(num_states, num_actions, num_steps, num_mc_runs, seed)
#experiment1.run()


def generate_data(num_data, ss, aa):
    A = 0.99
    B = 0.05
    C = 0.125
    var = 0.001
    V = 10 ** -5 * np.ones((ss, aa, ss))
    ra = np.ones(aa)
    y = np.ones(num_data)
    a = np.ones(num_data + 1)
    y[0] = 5.5
    for t in range(2, num_data - 1):
        a[t] = np.random.choice([a for a in range(aa)], 1)[0] + 1
        y[t] = A*y[t-1] + B*a[t] - C + B*np.random.normal(0, 1, 1)

    yy_len = (np.max(y) - np.min(y))/ss
    #print(yy_len)
    bins = np.linspace(np.min(y), np.max(y), ss)
    #print(bins)
    yy = np.digitize(y, bins)
    #plt.hist(yy)
    #plt.show()

    for t in range(1, num_data - 1):
        V[yy[t].astype(np.int64) - 1, a[t].astype(np.int64) - 1, yy[t - 1].astype(np.int64) - 1] = V[yy[t].astype(
            np.int64) - 1, a[t].astype(np.int64) - 1, yy[t - 1].astype(np.int64) - 1] + 1

    for at in range(aa):
        for s1 in range(ss):
            V[:, at, s1] = V[:, at, s1] / np.sum(V[:, at, s1])

        # pocets = np.zeros(ss)
        # poceta = np.zeros(aa)
        #
        # for j in range(ss):
        #     pocets[j] = np.sum(yy[:] == j)
        #
        # for k in range(aa):
        #     poceta[k] = np.sum(yy[:] == k)
    # with open("data_system", "wb") as f:
    #     pickle.dump(V, f, protocol=pickle.HIGHEST_PROTOCOL)
    return V

# FPD with Stopping using PE
num_states = 11
num_actions = 5
num_steps = 1
num_mc_runs = 3
horizon = 20
ideal_s = np.array([5, 6, 7])
ideal_a = np.array([3, 4])
w = 1.0
mu = 1.0
len_sim = 250

#number_data = 10000
#V = generate_data(number_data, 11, num_actions)
#print(V)
short_horizon = 50
experiment3 = Experiment(num_states, num_actions, num_steps, num_mc_runs, horizon, ideal_s, ideal_a, w, mu, len_sim)
experiment3.run()

