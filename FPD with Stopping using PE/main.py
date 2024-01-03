import numpy as np
from Learning_FPD_Stop_PE import Experiment

# FPD with Stopping using PE
num_states = 11
num_actions = 5
num_steps = 1
num_mc_runs = 25
horizon = 50
ideal_s = np.array([5, 6])
ideal_a = np.array([3])
w = 0.05
mu = 1.0
len_sim = 300
q = 0.85

if __name__ == '__main__':
    experiment3 = Experiment(num_states, num_actions, num_steps, num_mc_runs, horizon, ideal_s, ideal_a, w, mu, len_sim, q)
    experiment3.run()
    experiment3.store_results()

