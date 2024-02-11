import numpy as np
from scipy.optimize import fsolve
import random


class Agent:

    def __init__(self, num_states: int, num_actions: int, horizon: int, ideal_s: np.ndarray, ideal_a: np.ndarray,
                 w: np.float, mu: np.float, q: np.float) -> None:
        self.num_states, self.num_actions = num_states, num_actions
        self.horizon = horizon
        self.ideal_s, self.ideal_a = ideal_s, ideal_a
        self.w, self.mu = w, mu
        self.h = np.ones((num_states, horizon))
        self.d = np.ones((num_actions, num_states, horizon))
        self.r_io = np.ones((num_actions, num_states, horizon))
        self.t = horizon - 1
        self.r_o = np.ones((num_actions, num_states, horizon)) / self.num_actions
        self.init_r_o()
        self.model = np.ones((self.num_states, self.num_actions, self.num_states)) / self.num_states
        self.V = np.ones((num_states, num_actions, num_states))
        self.continues = 1
        self.history = np.array([1, 1])
        self.stop_time = horizon
        # Initial value selected based problem knowledge
        self.alpha = 1.2 * np.ones((num_actions, num_states))
        self.m_io = np.ones((num_states, num_actions, num_states)) / num_states
        self.m_io_long = np.ones((num_states, 2, num_actions, 2, num_states, 2)) / (num_states * 2)
        self.model_long = np.ones((num_states, 2, num_actions, 2, num_states, 2)) / (num_states * 2)
        self.dec_rule = np.ones((num_actions, 2, num_states, 2, horizon)) / (num_actions * 2)
        self.stop_rule = np.ones((2, num_states, 2)) / 2
        self.ideal_stop_rule = np.ones((2, num_states, 2)) / 2
        self.init_ideal_stop_rule(q)
        self.h_long = np.ones((num_states, 2, horizon))
        self.d_io_long = np.zeros((num_actions, 2, num_states, 2, horizon))

    def init_r_o(self):
        for ss in range(self.num_states):
            self.r_o[:, ss, self.t] = self.r_o[:, ss, self.t] / np.sum(self.r_o[:, ss, self.t])

    def init_model(self):
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for ss in range(self.num_states):
                    self.model[s, a, ss] = 1

    def init_ideal_stop_rule(self, q):
        for s_s in range(2):
            for s in range(self.num_states):
                for a_s in range(2):
                    if s_s == 0:
                        self.ideal_stop_rule[a_s, :, s_s] = 1 if a_s == s_s else 0
                    else:
                        self.ideal_stop_rule[a_s, :, s_s] = q if a_s == s_s else 1 - q

    def update_V(self, observed_state: np.int, observed_action: np.int, prev_state: np.int):
        """
        Updates statistics used for parametric estimation of the transition model.
        :param observed_state: state observed at time t
        :param observed_action: action at time t
        :param prev_state: state observed at time t-1
        :return:
        """
        self.V[observed_state, observed_action, prev_state] += 1

    def estimate_model(self) -> None:
        """
        Estimates transition probability of unknown Environment based on observed states and action stored in
        statistics/array self.V
        :return:
        """
        for a in range(self.num_actions):
            for s in range(self.num_states):
                self.model[:, a, s] = self.V[:, a, s] / np.sum(self.V[:, a, s])

    def stop(self):
        """
        Stops the DP
        :return:
        """
        self.continues = 0

    def predict_state(self, history) -> np.int:
        prob_cont = (self.model[:, history[0], history[1]]) / (np.sum(self.model[:, history[0], history[1]]))
        state = np.random.choice(np.arange(self.num_states), 1, p=prob_cont)[0]
        return state

    def generate_action(self, observed_state) -> int:
        """
        Generates new action and remember it
        :return: return new action
        """
        action = np.random.choice(np.arange(self.num_actions), 1, p=self.r_o[:, observed_state, self.t])[0]
        return action

    def evaluate_FPD(self) -> None:
        """
        Evaluates all values of FPD
        :return:
        """
        rho = np.ones((self.num_actions, self.num_states))
        self.t = self.horizon - 1
        for t in np.arange(self.horizon, 0, -1):
            rho, lam = self.evaluate_rho()
            for s in range(self.num_states):
                rho_max = np.max(rho[:, s])
            self.evaluate_d(rho, rho_max)
            # self.evaluate_m_io(lam)
            self.evaluate_r_io()
            self.normalize_r_io()
            self.evaluate_h()
            self.evaluate_r_o()
            self.normalize_r_o()
            self.t = t - 1

    def evaluate_PE(self) -> None:
        """
        Evaluates one step of PE in DP and m_io
        :return:
        """
        rho_max = np.zeros(self.num_states)
        rho, lam = self.evaluate_rho()
        for s in range(self.num_states):
            rho_max[s] = np.max(rho[:, s])
        self.evaluate_d(rho, rho_max)
        self.evaluate_m_io(lam)
        self.evaluate_r_io()
        self.normalize_r_io()
        self.evaluate_h()

    def evaluate_PE_short(self) -> None:
        """
        Evaluates one step of Preference Elicitation in DP in a standard way
        :return:
        """
        rho_max = np.zeros(self.num_states)
        rho, lam = self.evaluate_rho()
        for s in range(self.num_states):
            rho_max[s] = np.max(rho[:, s])
        self.evaluate_d(rho, rho_max)
        self.evaluate_r_io()
        self.normalize_r_io()
        self.evaluate_h()

    def evaluate_FPD_Stop(self):
        """
        Evaluates extended d, h and r_o
        :return:
        """
        self.evaluate_long_d_io()
        self.evaluate_long_h()
        self.evaluate_long_r_o()

    def evaluate_long_d_io(self):
        for a in range(self.num_actions):
            for a_s in range(2):
                for s in range(self.num_states):
                    for s_s in range(2):
                        # These forms are analytically obtained when designed formulas are insert into the extended theorem
                        if s_s == 0:
                            self.d_io_long[a, a_s, s, s_s, self.t] = 0
                        elif s_s == 1 and a_s == 0:
                            self.d_io_long[a, a_s, s, s_s, self.t] = 0
                        else:
                            for ss in range(self.num_states):
                                self.d_io_long[a, a_s, s, s_s, self.t] += self.model[ss, a, s] * np.log(
                                    self.model[ss, a, s] / (self.m_io[ss, a, s] * self.h_long[ss, 1, self.t]))

    def evaluate_long_h(self):
        aux_h = np.zeros((self.num_states, 2))
        for s in range(self.num_states):
            for s_s in range(2):
                if s_s == 0:
                    self.h_long[:, s_s, self.t - 1] = 1
                else:
                    for a in range(self.num_actions):
                        for a_s in range(2):
                            aux_h[s, s_s] += self.r_io[a, s, self.t] * self.ideal_stop_rule[a_s, s, s_s] * np.exp(
                                -self.d_io_long[a, a_s, s, s_s, self.t])
                    self.h_long[s, s_s, self.t - 1] = aux_h[s, s_s]

    def evaluate_long_r_o(self):
        for a in range(self.num_actions):
            for a_s in range(2):
                for s in range(self.num_states):
                    for s_s in range(2):
                        self.dec_rule[a, a_s, s, s_s, self.t] = self.r_io[a, s, self.t] * self.ideal_stop_rule[
                            a_s, s, s_s] * np.exp(-self.d_io_long[a, a_s, s, s_s, self.t]) / self.h_long[
                                                                    s, s_s, self.t - 1]

    def stop_decision(self) -> np.int:
        """
        DM function, where based on current self.dec_rule is a decision made if DP is stopped and current r_o is used
        or if we continue with DP
        :return:
        """
        # marginalization of probabilities in self.dec_rule
        np.random.seed()
        p_action = np.sum(self.dec_rule[:, :, self.history[1], self.continues, self.t], axis=1)
        p_stop = np.sum(self.dec_rule[:, :, self.history[1], self.continues, self.t], axis=0)
        has_nan1 = np.isnan(p_action)
        has_nan2 = np.isnan(p_stop)
        if np.any(has_nan1) or np.any(has_nan2):
            print("Problem")
        # generation of stopping action
        action = np.random.choice(np.arange(self.num_actions), 1, p=p_action)[0]
        stop_action = np.random.choice(np.arange(2), 1, p=p_stop)[0]
        # Check if we have reached horizon, if yes we stop the DP
        if self.t == 0:
            stop_action = 0
            self.stop_time = self.horizon - self.t
            self.stop()
        # If we have taken a stopping action
        if stop_action == 0:
            self.stop_time = self.horizon - self.t
            self.stop()
        self.t = self.t - 1
        return action

    def evaluate_r_o(self) -> None:
        """
        Evaluates
        :return:
        """
        for a in range(self.num_actions):
            for s in range(self.num_states):
                self.r_o[a, s, self.t] = np.exp(-(self.mu + 1) * self.d[a, s, self.t]) / self.h[s, self.t - 1]

    def evaluate_h(self) -> None:
        """
        Evaluates standard h function
        :return:
        """
        for s in range(self.num_states):
            self.h[s, self.t - 1] = np.sum(self.r_io[:, s, self.t] * np.exp(-self.d[:, s, self.t]))

    def evaluate_r_io(self) -> None:
        for a in range(self.num_actions):
            for s in range(self.num_states):
                self.r_io[a, s, self.t] = np.exp(-self.mu * self.d[a, s, self.t])

    def normalize_r_io(self):
        """
        Normalizes r_io
        :return:
        """
        for s in range(self.num_states):
            self.r_io[:, s, self.t] = self.r_io[:, s, self.t] / np.sum(self.r_io[:, s, self.t])

    def normalize_r_o(self):
        """
        Normalizes r_o
        :return:
        """
        for s in range(self.num_states):
            self.r_o[:, s, self.t] = self.r_o[:, s, self.t] / np.sum(self.r_o[:, s, self.t])

    def evaluate_m_io(self, lam: np.ndarray) -> None:
        """
        Evaluation of self.m_io based on theory
        :return:
        """
        for a in range(self.num_actions):
            for s in range(self.num_states):
                r_side = self.d[a, s, self.t] + np.sum(self.model[:, a, s] * np.log(self.h[:, self.t]))
                alpha = self.alpha[a, s]
                alpha = fsolve(self.opt_function, alpha, args=(lam, a, s, r_side))
                self.alpha[a, s] = alpha
                for s1 in range(self.num_states):
                    self.m_io[s1, a, s] = self.model[s1, a, s] * np.exp(-self.alpha[a, s] * self.model[s1, a, s])
        # normalization of m_io
        for a in range(self.num_actions):
            for s in range(self.num_states):
                self.m_io[:, a, s] = self.m_io[:, a, s] / np.sum(self.m_io[:, a, s])

    def evaluate_rho(self) -> (np.ndarray, np.ndarray):
        """
        Evaluates rho and lam
        :return:
        """
        rho = np.ones((self.num_actions, self.num_states))
        lam = np.ones((self.num_actions, self.num_states))
        aux_a = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            for vec in self.ideal_a:
                if vec == a:
                    aux_a[a] = 1
        for a in range(self.num_actions):
            for s in range(self.num_states):
                ideal_s_model = self.indicator_function(self.ideal_s, self.model[:, a, s])
                rho[a, s] = np.sum(ideal_s_model) + aux_a[a] * self.w
                lam[a, s] = np.sum(self.model[:, a, s] ** 2)
        return rho, lam

    def indicator_function(self, vectors: np.ndarray, arr: np.ndarray) -> np.ndarray:
        aux_arr = np.zeros((np.shape(arr)))
        for vec in range(np.shape(vectors)[0]):
            for i in range(np.shape(arr)[0]):
                if i == vectors[vec]:
                    aux_arr[i] = arr[i]
        return aux_arr

    def evaluate_d(self, rho: np.ndarray, rho_max: np.ndarray) -> None:
        """
        Evaluates d function
        :param rho:
        :param rho_max:
        :return:
        """
        d_max = self.aux_d(rho, rho_max)
        for a in range(self.num_actions):
            for ss in range(self.num_states):
                self.d[a, ss, self.t] = d_max[a, ss] + np.log(rho_max[ss] / rho[a, ss])

    def opt_function(self, alpha: np.array, lam: np.array, a: int, s: int,
                     r_side: np.array) -> np.array:
        l_side = alpha * lam[a, s] + np.log(np.sum(self.model[:, a, s] * np.exp(-alpha * self.model[:, a, s])))
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
        aux_d = np.zeros((self.num_actions, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for ss in range(self.num_states):
                    aux_d[a, ss] += self.model[s, a, ss] * np.log(
                        rho[a, ss] / (self.h[s, self.t] * rho_max[ss]))
        for a in range(self.num_actions):
            for ss in range(self.num_states):
                aux_d[a, ss] = max(0, np.max(aux_d[:, ss]))
        return aux_d


class Environment:

    def __init__(self, num_states: int, num_actions: int) -> None:
        self.num_states, self.num_actions = num_states, num_actions
        num_data = 10000
        self.model = self.init_model(num_data, self.num_states, self.num_actions)
        # self.model

    def init_model(self, num_data, ss, aa) -> np.ndarray:
        random.seed(5)
        A = 0.99
        B = 0.05
        C = 0.125
        V = 10 ** -5 * np.ones((ss, aa, ss))
        y = np.ones(num_data)
        a = np.ones(num_data + 1)
        y[0] = 5.5
        for t in range(1, num_data - 1):
            a[t] = np.random.choice([a for a in range(aa)], 1)[0] + 1
            y[t] = A * y[t - 1] + B * a[t] - C + B * np.random.normal(0, 1, 1)

        bins = np.linspace(np.min(y), np.max(y), ss)
        yy = np.digitize(y, bins)

        for t in range(1, num_data - 1):
            V[yy[t].astype(np.int64) - 1, a[t].astype(np.int64) - 1, yy[t - 1].astype(np.int64) - 1] = V[yy[t].astype(
                np.int64) - 1, a[t].astype(np.int64) - 1, yy[t - 1].astype(np.int64) - 1] + 1
        random.seed()
        for at in range(aa):
            for s1 in range(ss):
                V[:, at, s1] = V[:, at, s1] / np.sum(V[:, at, s1])
        return V

    def generate_state(self, action, prev_state, stop_state, past_history) -> np.int:
        # TODO: Create some model
        obs_state = np.random.choice(np.arange(self.num_states), 1, p=self.model[:, action, prev_state])[0]
        return obs_state.astype(int)
