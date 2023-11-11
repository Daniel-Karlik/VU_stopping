import numpy as np
from random import choices
import random

from numpy import ndarray

# Auxiliary vectors $v_{\M{a}}$ and $v_{\M{n}}$
NEIGHBOUR_VECTOR = [1, 1, 1, 1, 1, 3, 9, 2, 1]
AGENT_VECTOR = [1, 2, 9, 3, 1, 1, 1, 1, 1]
N_VEC = np.array([NEIGHBOUR_VECTOR], dtype=float)
A_VEC = np.array([AGENT_VECTOR], dtype=float)

SEED = 10
NUM_STATES = 3
NUM_ACTIONS = 4
dim = [NUM_STATES, NUM_ACTIONS, NUM_STATES]
history = [0, 0, 0]
HISTORY = np.array(history, dtype=int)


class Environment:

    def __init__(self, num_states, num_actions, seed):
        """
        Init an Environment representing unknown model consisting of transition matrix and
        history influencing upcoming state
        :param num_states: Number of states in estimated system
        :param num_actions: Number of actions in estimated system
        :param transition_matrix: Transition matrix represents probability matrix
        :param history: history[0] = previous state, history[1] = previous action
        variable t: represents the time stamp we are currently in
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.transition_matrix = self.initialize_transition_matrix(seed)
        self.time = 0

    def initialize_transition_matrix(self, seed: int) -> np.ndarray:
        """
        Initialize transition matrix of the Environment/system.
        This matrix is generated pseudorandomly using specified seed value.
        :param seed: seed for randomly generated matrix
        :returns self.transition_matrix
        """
        random.seed(seed, version=2)
        help_matrix = np.zeros((self.num_states, self.num_actions, self.num_states))
        for i in range(self.num_states):
            for j in range(self.num_actions):
                for k in range(self.num_states):
                    help_matrix[i, j, k] = random.random()
        # Section responsible for normalization
        for o in range(self.num_actions):
            for p in range(self.num_states):
                help_matrix[:, o, p] = help_matrix[:, o, p] / np.sum(help_matrix[:, o, p])
        return help_matrix

    def generate_state(self, cur_action: int, prev_state: int) -> int:
        """
        Generates new state according to the transition probability matrix and
        previously observed action and state
        """
        probabilities = self.transition_matrix[:, cur_action, prev_state]
        self.time += 1
        new_state = np.random.choice([s for s in range(self.num_states)], 1, p=probabilities)[0]
        return new_state


class BaseAgent:

    def __init__(self, num_states: int, num_actions: int):
        """
        Initialize agent class without information fusion.
        """
        self.num_states, self.num_actions = num_states, num_actions
        self.occurrence_table = self.initialize_prior_occurrence_table()
        self.stop_ind = 1
        self.history = HISTORY

    def update_occurrence_table(self, new_state: int) -> None:
        """
        Updates the occurrence table during each decision step, if we did not stopped
        """
        if self.stop_ind == 1:
            self.occurrence_table[new_state, self.history[1], self.history[0]] += 1

    def generate_action(self) -> int:
        """
        Generates new action and remember it
        :return: return new action
        """
        # TODO: Some more robust/clever solution
        new_action = np.random.choice([a for a in range(self.num_actions)], 1)[0]
        self.history[1] = new_action
        return new_action

    def make_prediction(self):
        """
        Makes a prediction based on occurence table
        :return:
        """
        # probabilities from occurence table
        probabilities = self.occurrence_table[:, self.history[1], self.history[0]] \
                        / np.sum(self.occurrence_table[:, self.history[1], self.history[0]])
        predicted_state = np.random.choice([a for a in range(self.num_states)], 1, p=probabilities)[0]
        # predicted_state = np.max(probabilities) # state with highest probability
        return predicted_state

    def initialize_prior_occurrence_table(self) -> np.ndarray:
        """
        Initialize prior occurrence table $V_{o}$.
        """
        return np.random.randint(1, 3, size=(self.num_states, self.num_actions, self.num_actions))
