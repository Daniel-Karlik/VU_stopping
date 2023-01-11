# This script is for testing "function for optimal stopping rule"

import numpy as np
import random

from matplotlib import pyplot as plt

# Preparation phase
num_states = 2
num_actions = 2

num_stop_states = 2
num_stop_actions = 2


SEED = 10
dim = [num_states, num_actions, num_states]

# Let's say that state 2 is more preferable than state 1

q_ideal = [0.1, 0.9]


def initialize_transition_matrix(seed: int) -> np.ndarray:
    """
    Initialize transition matrix.
    This matrix uniquely determines the environment.
    """
    random.seed(seed, version=2)
    help_matrix = np.ones((dim[0], dim[1], dim[2]))
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                help_matrix[i, j, k] = random.random()
    # Section responsible for normalization
    help_matrix = normalize(help_matrix)
    return help_matrix


def normalize(matrix: np.ndarray) -> np.ndarray:
    for o in range(dim[1]):
        for p in range(dim[2]):
            matrix[:, o, p] = matrix[:, o, p] / np.sum(matrix[:, o, p])
    return matrix


trans_matrix = initialize_transition_matrix(1)
print(trans_matrix)

print(trans_matrix[1, 0, 0])


# Testing phase


def change_matrix(matrix: np.ndarray, seed: int, variance: float) -> np.ndarray:
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                mu = matrix[i, j, k]
                random.seed(seed, version=2)
                matrix[i, j, k] = np.random.normal(mu, variance, 1)
    matrix = normalize(matrix)
    return matrix


random_matrix = change_matrix(trans_matrix, SEED, 0.1)
print(random_matrix)
HORIZONE = 10
d_storage = np.zeros((HORIZONE, num_actions, num_stop_actions, num_states, num_stop_states))
h_storage = np.zeros((HORIZONE, num_states, num_stop_states))

# Start of the backpropagation
h_init = 1
h_storage[-1] = h_init
# We initialize time index
t_init = HORIZONE - 1
t = t_init

model_matrix = random_matrix


def ideal_decision(action: int, prev_state: int) -> np.ndarray:
    # TODO fill some correct form of ideal decision rule
    output = np.random.uniform(size=3, low=0, high=1)
    sum_output = np.sum(output)
    return output[action]/sum_output


def model_prob(state: int, stop_state: int, action: int, stop_action: int,
               prev_state: int, prev_stop_state: int) -> float:
    """
    m(s_{t},\StSt_{t}|a_{t},\StAc_{t},s_{t-1},\StSt_{t-1})
    :param state:
    :param stop_state:
    :param action:
    :param stop_action:
    :param prev_state:
    :param prev_stop_state:
    :return:
    """
    if stop_action == stop_state:
        if prev_stop_state == 1:
            # if already did not stop the process
            return model_matrix[state, action, prev_state]
        else:
            # if the process is already stopped e.g. prev_stop_state == 0
            return 1 if prev_state == state else 0
    else:
        return 0


def model_ideal_prob(state: int, stop_state: int, action: int, stop_action: int,
                     prev_state: int, prev_stop_state: int) -> float:
    """
    m^{i}(s_{t},\StSt_{t}|a_{t},\StAc_{t},s_{t-1},\StSt_{t-1})
    :param state:
    :param stop_state:
    :param action:
    :param stop_action:
    :param prev_state:
    :param prev_stop_state:
    :return:
    """
    if stop_action == stop_state:
        if stop_action == 1:
            # if process should continue we return ideal model
            return trans_matrix[state, action, prev_state]
        else:
            # if the process should stop we don't care about ideal and let it be same as modelled
            if prev_stop_state == 1:
                # if already did not stop the process
                return model_matrix[state, action, prev_state]
            else:
                # if the process is already stopped e.g. prev_stop_state == 0
                return 1 if prev_state == state else 0
    else:
        return 0


def ideal_decision_rule(action: int, stop_action: int, prev_state: int,
                        prev_stop_state: int) -> np.ndarray:
    """
    Ideal decision rule as defined as in my text work
    :param action:
    :param stop_action:
    :param prev_state:
    :param prev_stop_state:
    :return: one value based on inputs
    """
    first_element = ideal_decision(action, prev_state) if stop_action == 1 \
        else 1/num_actions
    # as else value we used uniform action selection
    if prev_stop_state == 0:
        second_element = 1 if stop_action == 0 else 0
    else:
        second_element = q_ideal[prev_state] if stop_action == 1 else 1 - q_ideal[prev_state]

    final_output = first_element * second_element
    return final_output


def h_fun(time: int, state: int, stop_state: int) -> float: #or maybe -> np.ndarray?
    """
    Calculates h function values based on previous d function values and ideal decision rule
    :param time:
    :param state:
    :param stop_state:
    :return:
    """
    output = 0
    for i in range(num_actions):
        for stop in range(num_stop_actions):
            output += ideal_decision_rule(i, stop, state, stop_state) * \
                      np.exp(-d_storage[time, i, stop, state, stop_state])
    return output


def d_fun(time: int, action: int, stop_action: int, prev_state: int, prev_stop_state: int) -> float:
    """
    \mathsf{d}(a_{t},\StAc_{t},s_{t-1},\StSt_{t-1}) as defined in text
    :param time:
    :param action:
    :param stop_action:
    :param prev_state:
    :param prev_stop_state:
    :return:
    """
    output = 0
    for i in range(num_states):
        for stop in range(num_stop_states):
            model = model_prob(i, stop, action, stop_action, prev_state, prev_stop_state)
            id_mod = model_ideal_prob(i, stop, action, stop_action, prev_state, prev_stop_state)
            if model == 0:
                output += 0
            else:
                output += model*np.log(model/(id_mod*h_storage[time, i, stop]))
    return output


def normalize_d_fun() -> float:
    pass


# Script phase/Testing phase

for t_ind in range(t_init, -1, -1):
    if t_ind != t_init:
        for state in range(num_states):
            for stop_state in range(num_stop_states):
                h_storage[t_ind, state, stop_state] = h_fun(t_ind, state, stop_state)

    # range(start, stop, step)
    for action in range(num_actions):
        for stop_action in range(num_stop_actions):
            for state in range(num_states):
                for stop_state in range(num_stop_states):
                    d_storage[t_ind, action, stop_action, state, stop_state] = \
                        d_fun(t_ind, action, stop_action, state, stop_state)

    #print(h_storage[t_ind])
    #print(d_storage[t_ind])

print("h_storage is: ")
print(h_storage)
print("d_storage is: ")
print(d_storage)
# print(h_storage)










