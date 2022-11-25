# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import ParametricEstimator
from Environment import Environment
import random


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def normalize_trans_matrix(matrix):
    # Function for creating some random transition matrix according to selected dimension
    #
    n_s = np.shape(matrix)[0] # number of states
    n_a = np.shape(matrix)[1] # number of actions
    matrix_help = matrix
    for i in range(n_s):
        norm = np.sum(matrix[i, :])
        for j in range(n_a):
            matrix_help[i, j] = matrix[i, j] / norm

    return matrix_help

def model_dim(num_states, num_actions):
    #task = input("Enter dependencies of probability table: \n 'S' as state, 'A' as action, '-' as none")
    task = "SASA"
    task_to_array = [char for char in task]
    new_task = task_to_array
    ### This is the way how to force python to see string as a selected variable
    my_dict = {}
    x = "num_states"
    my_dict[x] = num_states
    y = "num_actions"
    my_dict[y] = num_actions
    ### -----------------------------
    for i in range(len(task_to_array)):
        new_task[i] = task_to_array[i].replace("S", "num_states")
        new_task[i] = task_to_array[i].replace("A", "num_actions")
        new_task[i] = task_to_array[i].replace("-", "")


    dimensions = np.ones(len(task_to_array))
    for i in range(len(task_to_array)):
        dimensions[i] = my_dict[task_to_array[i]]
    # dim represents shape of the new matrix
    dim = dimensions.astype(int)
    return dim

def init_3d_trans_matrix(dimensions, seed):
    '''
    Init_3d_trans_matrix creates randomly 3D random transition matrix
    :param dimensions: should be len == 3
    :param seed: initialize random matrix
    :return: transition matrix
    '''
    dim = dimensions
    random.seed(seed, version=2)
    help_matrix = np.ones((dim[0], dim[1], dim[2]))
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                help_matrix[i, j, k] = random.random()

    for o in range(dim[1]):
        for p in range(dim[2]):
            help_matrix[:, o, p] = help_matrix[:, o, p]/np.sum(help_matrix[:, o, p])

    return help_matrix


# Leave this to spare time and future work...
def cartesian_product(matrix, vector):
    size = np.shape(matrix)[0]
    vec_size = np.shape(vector)[0]
    n = len(np.shape(matrix))
    help = np.ones((size*vec_size, n+1))
    index = 0
    for i in range(size):
        for k in range(vec_size):
            help[index] = np.append(matrix[i], vector[k])
            index = index + 1

    return help


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    # =============================================
    # INITIAL COMMENT
    # I would like to split my testing task into 2 different tasks.
    # First is stopping the parametric estimation designed in this
    # file and the second is the Secretary Problem in other file.
    # =============================================
    # Here the code continues

    #primary_est = ParametricEstimator(num_states, num_actions, 3)
    qq = model_dim(3, 2)
    print(qq)
    print(np.shape(qq))
    matrix = init_3d_trans_matrix([3, 4, 3], 10)
    print("matrix [2, 2, 2] is : ")
    print(matrix)
    print("First row")
    print(matrix[:, 0, 0])
    #print(matrix[:, 0, 1])
    num_states = 3
    num_actions = 4
    seed = 10
    env_matrix = init_3d_trans_matrix([num_states, num_actions, num_states], seed)
    history = [2, 2]
    env = Environment(num_states, num_actions, env_matrix, history)
    print("Environment matrix")
    print(env.transition_matrix)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
