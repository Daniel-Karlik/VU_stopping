# This is main script for running examples related with Daniel Karlik's Master Thesis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Environment import Environment, BaseAgent
from experiment import Experiment

# Defining the task
num_states = 5
num_actions = 7
num_steps = 10
num_mc_runs = 10
seed = 10

# function experiment
experiment1 = Experiment(num_states, num_actions, num_steps, num_mc_runs, seed)
experiment1.run()

# FPD with Stopping using PE
num_states = 2
num_actions = 2
horizon = 10
ideal_s = np.array([[0, 1], [1, 1]])
ideal_a = np.array([0, 1])
w = 1
mu = 1

experiment2 = Experiment_FPD(num_states, num_actions, horizon, ideal_s, ideal_a, w, mu)

