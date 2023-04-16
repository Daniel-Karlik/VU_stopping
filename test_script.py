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


# Testing phase


def change_matrix(matrix: np.ndarray, seed: int, variance: float) -> np.ndarray:
    matrix_help = matrix.copy()
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                mu = matrix_help[i, j, k]
                random.seed(seed, version=2)
                matrix_help[i, j, k] = np.random.normal(mu, variance, 1)
    matrix_out = normalize(matrix_help)
    return matrix_out


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

id_dec = np.random.uniform(size=(num_states, num_actions), low=0, high=1)
id_dec[0] = id_dec[0]/np.sum(id_dec[0])
id_dec[1] = id_dec[1]/np.sum(id_dec[1])
# print("id_dec")
# print(id_dec)

print("trans matrix")
print(trans_matrix)

def ideal_decision(action: int, prev_state: int) -> float:
    # TODO fill some correct form of ideal decision rule
    return id_dec[prev_state, action]


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
        if stop_state == 1:
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
            # if prev_stop_state == 1:
            #     # if already did not stop the process
            #     return model_matrix[state, action, prev_state]
            # else:
            #     # if the process is already stopped e.g. prev_stop_state == 0
            #     return 1 if prev_state == state else 0
            if stop_state == 1:
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
                      np.exp(-d_storage[time+1, i, stop, state, stop_state])
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
            # print("regular model and id_model")
            # print(i, " stop_state is ", stop, " ", action, " stop_action is ", stop_action, " ", prev_state,
            #       " prev_stop_state is ", prev_stop_state)
            # print(i, stop, action, stop_action, prev_state, prev_stop_state)
            # print(model)
            id_mod = model_ideal_prob(i, stop, action, stop_action, prev_state, prev_stop_state)
            # print(id_mod)
            if model == 0:
                output += 0
            else:
                output += model*np.log(model/(id_mod*h_storage[time, i, stop]))
    fun_output = output
    return fun_output


def normalize_d_fun() -> float:
    pass


def optimal_policy(num_actions: int, num_stop_actions: int, num_states: int, num_stop_states: int, h_stored: np.ndarray,
                   d_stored: np.ndarray) -> np.ndarray:
    t_len = np.shape(h_stored)[0]
    opt_policy = np.zeros((t_len, num_actions, num_stop_actions, num_states, num_stop_states))
    for t in range(t_len-1, -1, -1):
        for action in range(num_actions):
            for stop_action in range(num_stop_actions):
                for state in range(num_states):
                    for stop_state in range(num_stop_states):
                        opt_policy[t, action, stop_action, state, stop_state] = \
                            ideal_decision_rule(action, stop_action, state, stop_state) * \
                            (np.exp(-d_stored[t, action, stop_action, state, stop_state]) /
                             h_stored[t, state, stop_state])
    return opt_policy



# Script phase/Testing phase
print("trans matrix")
print(trans_matrix)

for t_ind in range(t_init, -1, -1):
    if t_ind != t_init:
        for state in range(num_states):
            for stop_state in range(num_stop_states):
                h_storage[t_ind, state, stop_state] = h_fun(t_ind, state, stop_state) # h_fun(t_ind+1, state, stop_state)

    # range(start, stop, step)
    for action in range(num_actions):
        for stop_action in range(num_stop_actions):
            for state in range(num_states):
                for stop_state in range(num_stop_states):
                    d_storage[t_ind, action, stop_action, state, stop_state] = \
                        d_fun(t_ind, action, stop_action, state, stop_state)

    print(" \n new t_ind = ", t_ind, "\n")

    #print(h_storage[t_ind])
    #print(d_storage[t_ind])

print("h_storage is: ")
print(h_storage)
print("d_storage is: ")
print(d_storage)


print("OPTIMAL POLICY EVALUATION")
opt_policy = optimal_policy(num_actions, num_stop_actions, num_states, num_stop_states, h_storage, d_storage)
print(opt_policy)

print("IDEAL DECISION RULE")
id_dec_rule = opt_policy = np.zeros((num_actions, num_stop_actions, num_states, num_stop_states))
for action in range(num_actions):
    for stop_action in range(num_stop_actions):
        for state in range(num_states):
            for stop_state in range(num_stop_states):
                # print(action, stop_action, state, stop_state)
                # print(ideal_decision_rule(action, stop_action, state, stop_state))
                id_dec_rule[action, stop_action, state, stop_state] = ideal_decision_rule(action, stop_action, state,
                                                                                          stop_state)
print(id_dec_rule)

# Zda se ze ideal_decision_rule funguje spravne
# print("Checking state and stop_state")
# print(id_dec_rule[:, :, 0, 1])
# print(h_storage)


# auxiliary part


def aux_d(action: int, prev_state: int) -> float:
    output = 0
    for state in range(num_states):
        for stop in range(num_stop_states):
            model = model_matrix[state, action, prev_state]
            print("regular model and id_model")
            # print(i, stop, action, stop_action, prev_state, prev_stop_state)
            print(model)
            id_mod = trans_matrix[state, action, prev_state]
            print(id_mod)
            if model == 0:
                output += 0
            else:
                output += model*np.log(model/id_mod)
    return output


# h_out = np.zeros((2, 2))
# for state in range(num_states):
#     for stop_state in range(num_stop_states):
#         for action in range(num_actions):
#             for stop_action in range(num_stop_actions):
#                 id_dr = ideal_decision_rule(action, stop_action, state, stop_state)
#                 # print(id_dr)
#                 # print(d_storage[-1, action, stop_action, state, stop_state])
#                 h_out[state, stop_state] += id_dr * np.exp(-d_storage[-1, action, stop_action, state, stop_state])
# print(h_out)

a = 0

# print("d as from theory:")
# for action in range(num_actions):
#     for prev_state in range(num_states):
#         print(aux_d(action, prev_state))
#
# print("model_matrix")
# print(model_matrix)
# print("real matrix")
# print(trans_matrix)










