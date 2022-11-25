import numpy as np


class ParametricEstimator:

    def __init__(self, num_states, num_actions, dimension):
        """
        Init an empty Parametric Estimator
        :param num_states: Number of states in estimated system
        :param num_actions: Number of actions in estimated system
        :param dimension: Number of dimension of estimated transition matrix
        variable statistics: Represents the observations observed or from prior knowledge
        variable t: represents the time stamp we are currently in
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.statistics = []
        self.dimension = dimension
        self.time = 0

    def print_statistics(self):
        print(self.statistics)

    def update_observation(self, observation):
        self.statistics[observation] += 1

