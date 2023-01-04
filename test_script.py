# This script is for testing "function for optimal stopping rule"

import numpy as np
import random

from matplotlib import pyplot as plt

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


def change_matrix(matrix: np.ndarray, seed: int, variance: int) -> np.ndarray:
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












