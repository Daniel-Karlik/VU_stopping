# This script is for testing "function for optimal stopping rule"

import numpy as np
import random
from scipy.optimize import fsolve

from matplotlib import pyplot as plt


class FPD_Stop_PE:

    def __init__(self, num_states: int, num_actions: int, horizon: int, ideal_s: np.ndarray, ideal_a: np.ndarray,
                 w: np.float, mu: np.float) -> None:
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
            self.normalize_r_o()
            self.t = t - 1

    def evaluate_r_o(self) -> None:
        for a in range(self.num_actions):
            for a_s in range(2):
                for s in range(self.num_states):
                    for s_s in range(2):
                        self.r_o[a, a_s, s, s_s, self.t] = np.exp(-(self.mu + 1) * self.d[a, a_s, s, s_s, self.t]) / \
                                                           self.h[s, s_s, self.t - 1]

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

    def normalize_r_o(self):
        """
        Normalizes r_o
        :return:
        """
        for s in range(self.num_states):
            for s_s in range(2):
                self.r_o[:, :, s, s_s, self.t] = self.r_o[:, :, s, s_s, self.t] / np.sum(
                    self.r_o[:, :, s, s_s, self.t])
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


class Agent:

    def __init__(self, ss: int, aa: int, horizon: np.int, w: int, s0: int, nu: int, si: int, ai: np.ndarray,
                 alfa: np.ndarray,
                 sigma: int) -> None:
        self.num_states, self.num_actions = num_states, num_actions
        self.horizon = horizon
        self.r_io = np.ones((num_actions, 2, num_states, 2, horizon))
        self.r_o = np.ones((num_actions, 2, num_states, 2, horizon))
        self.model = np.ones((self.num_states, 2, self.num_actions, 2, self.num_states, 2)) / (self.num_states * 2)
        self.generate_model()
        self.t = t
        # TODO: Is this shape OK?
        self.alpha = 1.2 * np.ones((num_actions, 2, num_states, 2))
        self.m_io = np.ones((num_states, 2, num_actions, 2, num_states, 2)) / (num_states * 2)
        self.ss = ss
        self.aa = aa
        self.horizon = horizon
        self.w = w
        self.s0 = s0
        self.nu = nu
        self.si = si
        self.ai = ai
        self.model = self.create_model()
        self.mi = np.ones((self.ss, self.aa, self.ss)) / self.ss
        self.ri = np.ones((self.aa, self.ss)) / self.aa
        self.r = np.ones((self.aa, self.ss)) / self.aa
        self.V = np.ones((self.ss, self.aa, self.ss))

    def create_model(self):

        model = np.ones(tuple([self.ss, self.aa, self.ss], ))

        for s1 in range(self.ss):
            for a in range(self.aa):
                for s2 in range(self.ss):
                    model[s2, a, s1] = np.exp(-(s2 - s1 - a) ** 2 / (2 * self.sigma ** 2))

        model = self.normalize_proba_model(model)

        return model

    def normalize_proba_model(self, model):
        for s1 in range(self.ss):
            for a in range(self.aa):
                model[:, a, s1] = model[:, a, s1] / np.sum(model[:, a, s1])

        return model

    def learn(self, data2):
        s = data2.states[data2.t]
        a = data2.actions[data2.t - 1]
        s1 = data2.states[data2.t - 1]
        self.V[s.astype(np.int64), a.astype(np.int64), s1.astype(np.int64)] = self.V[s.astype(np.int64), a.astype(
            np.int64), s1.astype(np.int64)] + 1

        self.model[:, a.astype(np.int64), s1.astype(np.int64)] = self.V[:, a.astype(np.int64),
                                                                 s1.astype(np.int64)] / np.sum(
            self.V[:, a.astype(np.int64), s1.astype(np.int64)])

    def opt_mi(self, a, s1):

        for s in range(self.ss):
            self.mi[s, a, s1] = self.model[s, a, s1] * np.exp(-self.alfa[a, s1] * self.model[s, a, s1])

    def opt_mio(self, a, s1, o):

        for s in range(self.ss):
            self.mi[s, a, s1] = np.exp(-self.alfa[a, s1] * o[s, a, s1])

    def opt_ri(self, s1, d):
        for a in range(self.aa):
            self.ri[a, s1] = np.exp(-self.nu * d[a, s1])

        self.ri[:, s1] = self.ri[:, s1] / np.sum(self.ri[:, s1])

    def opt_function(self, alfa_var, lambda_var, a, s1, r_side):
        l_side = alfa_var * lambda_var[a, s1] + np.log(
            np.sum(self.model[:, a, s1] * np.exp(-alfa_var * self.model[:, a, s1])))
        f = l_side - r_side

        return f

    def opto_function(self, alfa_var, a, s1, r_side, o):
        l_side = np.log(np.sum(np.exp(-alfa_var * o[:, a, s1])) / self.ss)

        f = l_side - r_side

        return f

    def calculate_alfa(self):
        lambda_var = np.zeros((self.aa, self.ss))
        rho = np.zeros((self.aa, self.ss))
        rho_max = np.zeros(self.ss)
        for s1 in range(self.ss):
            for a in range(self.aa):
                lambda_var[a, s1] = np.sum(self.model[:, a, s1] ** 2)
                rho[a, s1] = np.sum(self.model[self.si, a, s1])
                if np.any(self.ai == a):
                    rho[a, s1] = (1 - self.w) * rho[a, s1] + self.w
                if rho[a, s1] >= rho_max[s1]:
                    rho_max[s1] = rho[a, s1]

        for tau in range(self.h, 0, -1):
            d = np.zeros((self.aa, self.ss))
            d_opt = np.zeros((self.aa, self.ss))
            d_help = np.zeros(self.ss)
            for s1 in range(self.ss):
                var = np.zeros(self.aa)
                for a in range(self.aa):
                    var[a] = np.sum(self.model[:, a, s1] * np.log(rho[a, s1] / (rho_max[s1] * self.gam[:])))

                    if d_help[s1] < var[a]:
                        d_help[s1] = var[a]
                if d_help[s1] < 0:
                    d_help[s1] = 0

                for a in range(self.aa):
                    d_opt[a, s1] = d_help[s1] + np.log(rho_max[s1] / rho[a, s1])
                    if np.sum(self.model[:, a, s1] != np.ones(self.ss)) > 0:
                        r_side = d_opt[a, s1] + np.sum(self.model[:, a, s1] * np.log(self.gam[:]))

                        alfaa = self.alfa[a, s1]
                        alfaa = fsolve(self.opt_function, alfaa, args=(lambda_var, a, s1, r_side))
                        self.alfa[a, s1] = alfaa
                        self.opt_mi(a, s1)
                    else:
                        o = np.ones((self.ss, self.aa, self.ss))
                        o[self.ss, :, :] = - (self.ss - 1) * o[self.ss, self.aa, self.ss]
                        o = o / self.aa

                        r_side = d_opt[a, s1] + np.sum(
                            self.model[:, a, s1] * np.log((self.gam[:] * rho_max[s1]) / rho[a, s1]))

                        alfaa = self.alfa[a, s1]
                        alfaa = fsolve(self.opto_function, alfaa, args=(lambda_var, a, s1, r_side, o))
                        self.alfa[a, s1] = alfaa
                        self.opt_mio(a, s1, o)

                for a in range(self.aa):
                    self.mi[:, a, s1] = self.mi[:, a, s1] / np.sum(self.mi[:, a, s1])

                for a in range(self.aa):
                    d[a, s1] = np.sum(
                        self.model[:, a, s1] * np.log(self.model[:, a, s1] / (self.gam[:] * self.mi[:, a, s1])))

                self.opt_ri(s1, d_opt)
                self.gam[s1] = np.sum(self.ri[:, s1] * np.exp(-d_opt[:, s1]))
                self.r[:, s1] = (self.ri[:, s1] * np.exp(-d_opt[:, s1])) / self.gam[s1]
