# This script is for testing "function for optimal stopping rule"

import numpy as np
import random
from scipy.optimize import fsolve

from matplotlib import pyplot as plt


class FPD_Stop_PE:

    def __init__(self, num_states: int, num_actions: int, horizon: int, ideal_s: np.ndarray, ideal_a: np.ndarray,
                 w: int, mu: int) -> None:
        """
        Initialize an evaluation process of FPD with stopping using PE
        :param num_states:
        :param num_actions:
        :param horizon:
        :param ideal_s: set of ideal states
        :param ideal_a: set o ideal actions
        :param w: weight of importance between actions and states
        :param mu: selecting exploring parameter
        """
        self.num_states, self.num_actions = num_states, num_actions
        self.horizon = horizon
        self.ideal_s, self.ideal_a = ideal_s, ideal_a
        self.w, self.mu = w, mu
        self.h = np.ones((num_states, 2, horizon))
        self.d = np.ones((num_actions, 2, num_states, 2, horizon))
        self.r_io = np.ones((num_actions, 2, num_states, 2, horizon))
        self.r_o = np.ones((num_actions, 2, num_states, 2, horizon))
        self.model = np.ones((self.num_states, 2, self.num_actions, 2, self.num_states, 2)) / (self.num_states * 2)
        self.generate_model()
        self.model_short = help_matrix = np.ones((self.num_states, self.num_actions, self.num_states)) / self.num_states
        self.generate_short_model()
        self.t = horizon - 1
        # TODO: Is this shape OK?
        self.alpha = 1.2 * np.ones((num_actions, 2, num_states, 2))
        self.m_io = np.ones((num_states, 2, num_actions, 2, num_states, 2)) / (num_states * 2)

    def generate_model(self):
        """
        Randomly generated model
        :return:
        """
        random.seed(2)
        help_matrix = np.ones((self.num_states, 2, self.num_actions, 2, self.num_states, 2))
        for i in range(self.num_states):
            for j in range(2):
                for k in range(self.num_actions):
                    for l in range(2):
                        for m in range(self.num_states):
                            for n in range(2):
                                help_matrix[i, j, k, l, m, n] = random.random()

        for k in range(self.num_actions):
            for l in range(2):
                for m in range(self.num_states):
                    for n in range(2):
                        help_matrix[:, :, k, l, m, n] = help_matrix[:, :, k, l, m, n] / np.sum(
                            np.sum(help_matrix[:, :, k, l, m, n], axis=1), axis=0)
        self.model = help_matrix

    def generate_short_model(self):
        """
        Generates short model of the environment
        :return:
        """
        random.seed(2)
        help_matrix = np.ones((self.num_states, self.num_actions, self.num_states))
        for i in range(self.num_states):
            for j in range(self.num_actions):
                for k in range(self.num_states):
                    help_matrix[i, j, k] = random.random()

        for o in range(self.num_actions):
            for p in range(self.num_states):
                help_matrix[:, o, p] = help_matrix[:, o, p] / np.sum(help_matrix[:, o, p])
        self.model_short = help_matrix

    def evaluate_FPD(self) -> None:
        """
        Evaluates all values of FPD
        :return:
        """
        rho = np.ones((self.num_actions, 2, self.num_states, 2))
        for t in np.arange(self.horizon, 0, -1):
            rho, lam = self.evaluate_rho()
            for s in range(self.num_states):
                for s_s in range(2):
                    rho_max = np.max(np.max(rho[:, :, s, s_s], axis=1), axis=0)
            self.evaluate_d(rho, rho_max)
            # self.evaluate_m_io(lam)
            self.evaluate_r_io()
            self.normalize_r_io()
            self.evaluate_h()
            self.evaluate_r_o()
            self.t = t - 1

    def evaluate_r_o(self) -> None:
        for a in range(self.num_actions):
            for a_s in range(2):
                for s in range(self.num_states):
                    for s_s in range(2):
                        self.r_o[a, a_s, s, s_s, self.t] = (-(self.mu + 1) * self.d[a, a_s, s, s_s, self.t]) / self.h[
                            s, s_s, self.t - 1]

    def evaluate_h(self) -> None:
        """
        Evaluates h function
        :return:
        """
        for s in range(self.num_states):
            for s_s in range(2):
                self.h[s, s_s, self.t - 1] = np.sum(np.sum(self.r_io[:, :, s, s_s, self.t] * np.exp(
                            -self.d[:, :, s, s_s, self.t]), axis=0), axis=0)

    def evaluate_r_io(self) -> None:
        for a in range(self.num_actions):
            for a_s in range(2):
                for s in range(self.num_states):
                    for s_s in range(2):
                        self.r_io[a, a_s, s, s_s, self.t] = np.exp(-self.mu * self.d[a, a_s, s, s_s, self.t])

        for a in range(self.num_actions):
            for a_s in range(2):
                for s in range(self.num_states):
                    for s_s in range(2):
                        self.r_io[a, a_s, s, s_s, self.t] = self.r_io[a, a_s, s, s_s, self.t] / (np.sum(
                            np.sum(self.r_io[:, :, s, s_s, self.t], axis=0), axis=0))

    def normalize_r_io(self):
        """
        Normalizes r_io
        :return:
        """
        for s in range(self.num_states):
            for s_s in range(2):
                self.r_io[:, :, s, s_s, self.t] = self.r_io[:, :, s, s_s, self.t] / np.sum(
                            self.r_io[:, :, s, s_s, self.t])

    def evaluate_m_io(self, lam: np.ndarray) -> None:
        """
        If self.model_short is non-uniform, then...
        :return:
        """
        for a in range(self.num_actions):
            for a_s in range(2):
                for s in range(self.num_states):
                    for s_s in range(2):
                        # TODO: solve if I need long d or short d HERE
                        r_side = self.d[a, a_s, s, s_s, self.t] + np.sum(np.sum(
                            self.model[:, :, a, a_s, s, s_s] * np.log(self.h[:, :, self.t]), axis=1), axis=0)
                        alpha = self.alpha[a, a_s, s, s_s]
                        # l_side = alpha * lam[a, ss] + np.log(np.sum(self.model[:, a, ss] * np.exp(-alpha * self.model[:, a, ss])))
                        alpha = fsolve(self.opt_function, alpha, args=(lam, a, a_s, s, s_s, r_side))
                        self.alpha[a, a_s, s, s_s] = alpha
                        for s1 in range(self.num_states):
                            for s_s1 in range(2):
                                self.m_io[s1, s_s1, a, a_s, s, s_s] = np.exp(
                                    -self.alpha[a, a_s, s, s_s] * self.model[s1, s_s1, a, a_s, s, s_s])

    def evaluate_rho(self) -> (np.ndarray, np.ndarray):
        """
        Evaluate rho
        :return:
        """
        rho = np.ones((self.num_actions, 2, self.num_states, 2))
        lam = np.ones((self.num_actions, 2, self.num_states, 2))
        for a in range(self.num_actions):
            for a_s in range(2):
                for s in range(self.num_states):
                    for s_s in range(2):
                        rho[a, a_s, s, s_s] = np.sum(
                            np.sum(self.ideal_s * self.model[:, :, a, a_s, s, s_s] + self.ideal_a * self.w, axis=1),
                            axis=0)
                        lam[a, s] = np.sum(np.sum(self.model[:, :, a, a_s, s, s_s] ** 2, axis=1), axis=0)
        return rho, lam

    def evaluate_d(self, rho: np.ndarray, rho_max: np.ndarray) -> None:
        """
        Evaluates d function
        :param rho:
        :param rho_max:
        :return:
        """
        d_max = np.ones((np.shape(self.d)))
        d_max = self.aux_d(rho, rho_max)
        self.d[:, :, :, :, self.t] = d_max + np.log(rho_max / rho)

    def opt_function(self, alpha: np.array, lam: np.array, a: int, a_s: int, s: int, s_s: int,
                     r_side: np.array) -> np.array:
        l_side = alpha * lam[a, a_s, s, s_s] + np.log(
            np.sum(np.sum(self.model[:, :, a, a_s, s, s_s] * np.exp(-alpha * self.model[:, :, a, a_s, s, s_s]),
                          axis=1), axis=0))
        f = l_side - r_side
        return f

    def aux_d(self, rho: np.array, rho_max: np.array) -> np.ndarray:
        """
        Auxiliary evaluation for d function
        :param rho:
        :param rho_max:
        :param t:
        :return:
        """
        aux_d = np.zeros((self.num_actions, 2, self.num_states, 2))
        for s in range(self.num_states):
            for s_s in range(2):
                for a in range(self.num_actions):
                    for a_s in range(2):
                        for ss in range(self.num_states):
                            for ss_s in range(2):
                                aux_d[a, a_s, ss, ss_s] += np.sum(self.model[s, s_s, a, a_s, ss, ss_s] * np.log(
                                    rho[a, a_s, ss, ss_s] / (self.h[s, s_s, self.t] * rho_max)))
        for a in range(self.num_actions):
            for a_s in range(2):
                for ss in range(self.num_states):
                    for ss_s in range(2):
                        aux_d[a, a_s, ss, ss_s] = max(0, np.max(aux_d[:, :, ss, ss_s]))
        return aux_d
