import numpy as np
from scipy.optimize import fsolve


class Agent:

    def __init__(self, num_states: int, num_actions: int, horizon: int, ideal_s: np.ndarray, ideal_a: np.ndarray,
                 w: np.float, mu: np.float) -> None:
        self.num_states, self.num_actions = num_states, num_actions
        self.horizon = horizon
        self.ideal_s, self.ideal_a = ideal_s, ideal_a
        self.w, self.mu = w, mu
        self.h = np.ones((num_states, 2, horizon))
        self.d = np.ones((num_actions, 2, num_states, 2, horizon))
        self.r_io = np.ones((num_actions, 2, num_states, 2, horizon))
        self.t = horizon - 1
        self.r_o = np.ones((num_actions, 2, num_states, 2, horizon))/(self.num_actions*2)
        self.init_r_o()
        self.model = np.ones((self.num_states, 2, self.num_actions, 2, self.num_states, 2)) / (self.num_states * 2)
        self.init_model()
        self.V = np.ones((num_states, 2, num_actions, 2, num_states, 2))
        self.continues = 1
        self.history = np.array([[1, 1], [1, 1]])
        # TODO: Is this shape OK?
        # self.alpha = 1.2 * np.ones((num_actions, 2, num_states, 2))
        # self.m_io = np.ones((num_states, 2, num_actions, 2, num_states, 2)) / (num_states * 2)

    def init_r_o(self):
        for a in range(self.num_actions):
            for a_s in range(2):
                for ss in range(self.num_states):
                    for ss_s in range(2):
                        if ss_s != a_s:
                            self.r_o[a, a_s, ss, ss_s, self.t] = 0
                        else:
                            self.r_o[a, a_s, ss, ss_s, self.t] = 1
        for ss in range(self.num_states):
            for ss_s in range(2):
                self.r_o[:, :, ss, ss_s, self.t] = self.r_o[:, :, ss, ss_s, self.t] / (np.sum(
                    np.sum(self.r_o[:, :, ss, ss_s, self.t], axis=0), axis=0))

    def init_model(self):
        for s in range(self.num_states):
            for s_s in range(2):
                for a in range(self.num_actions):
                    for a_s in range(2):
                        for ss in range(self.num_states):
                            for ss_s in range(2):
                                if s_s == 0:
                                    self.model[s, s_s, a, a_s, ss, ss_s] = 1 if s == ss else 0
                                elif s_s != a_s or a_s != ss_s:
                                    self.model[s, s_s, a, a_s, ss, ss_s] = 0
                                else:
                                    self.model[s, s_s, a, a_s, ss, ss_s] = 1

        for a in range(self.num_actions):
            for a_s in range(2):
                for s in range(self.num_states):
                    for s_s in range(2):
                        self.model[:, :, a, a_s, s, s_s] = self.model[:, :, a, a_s, s, s_s] / (np.sum(
                            np.sum(self.model[:, :, a, a_s, s, s_s], axis=0), axis=0))

    def update_V(self, observed_state: np.int, observed_action: np.int, prev_state: np.int):
        """
        Updates statistics used for parametric estimation of the transition model.
        :param observed_state:
        :param observed_action:
        :param prev_state:
        :return:
        """
        if self.continues == 1:
            self.V[observed_state, 1, observed_action, 1, prev_state, 1] += 1
        else:
            self.V[observed_state, 0, observed_action, 0, prev_state, 0] += 1
        #self.history[0, 0] = observed_action

    def estimate_model(self) -> None:
        for a in range(self.num_actions):
            for a_s in range(2):
                for s in range(self.num_states):
                    for s_s in range(2):
                        self.V[:, :, a, a_s, s, s_s] = self.V[:, :, a, a_s, s, s_s] / (np.sum(
                            np.sum(self.V[:, :, a, a_s, s, s_s], axis=0), axis=0))

    def stop(self):
        self.continues = 0

    def predict_state(self, history) -> np.int:
        # If the process continues we generate actions normally otherwise randomly
        if self.continues == 1:
            prob_cont = (self.model[:, 1, history[0, 0], 1, history[1, 0], 1])/(np.sum(self.model[:, 1, history[0, 0], 1, history[1, 0], 1]))
            #state = np.random.choice(np.arange(self.num_states), 1, p=prob_cont)[0]
            state = prob_cont.argmax(axis=0)
            #state = np.random.choice(np.arange(2*self.num_states), 1, p=self.model[:, :, history[0, 0], 1, history[1, 0], 1].ravel())[0]
            # we have flattened 2D probability and we need to predict state and stop_state
            #predicted_state = np.array([(state - (state % 2))/2, state % 2])
            predicted_state = np.array([state, 1])
        else:
            state = history[1, 0]
            #state = np.random.choice([s for s in range(self.num_states)], 1)[0]
            predicted_state = np.array([state, 0])
        return predicted_state

    def generate_action(self, observed_state, time) -> int:
        """
        Generates new action and remember it
        :return: return new action
        """
        if self.continues == 1:
            action = np.random.choice(np.arange(2 * self.num_actions), 1,  p=self.r_o[:, :, observed_state, 1, self.t].ravel())[0]
            # we have flattened 2D probability and we need to predict state and stop_state
            new_action = np.array([(action - (action % 2)) / 2, action % 2])
        else:
            action = np.random.choice([a for a in range(self.num_actions)], 1)[0]
            new_action = np.array([action, 0])
        #self.history[0, :] = new_action
        # We dont stop too early
        if time < 30:
            new_action[1] = 1
        if new_action[1] == 0:
            self.stop()
        return new_action

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
                    for s_s in np.arange(2):
                        ideal_s_model = self.indicator_function(self.ideal_s, self.model[:, :, a, a_s, s, s_s])
                        rho[a, a_s, s, s_s] = np.sum(np.sum(ideal_s_model + self.ideal_a * self.w, axis=1), axis=0)
                        lam[a, a_s, s, s_s] = np.sum(np.sum(self.model[:, :, a, a_s, s, s_s] ** 2, axis=1), axis=0)
                        #print(a, a_s, s, s_s)
        return rho, lam

    def indicator_function(self, vectors: np.ndarray, arr: np.ndarray) -> np.ndarray:
        aux_arr = np.zeros((np.shape(arr)))
        for vec in range(np.shape(vectors)[0]):
            for i in range(np.shape(arr)[0]):
                for j in range(np.shape(arr)[1]):
                    if i == vectors[vec, 0] and vectors[vec, 1] == j:
                        aux_arr[i, j] = arr[i, j]
        return aux_arr

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


class Environment:

    def __init__(self, num_states: int, num_actions: int) -> None:
        self.num_states, self.num_actions = num_states, num_actions
        num_data = 10000
        self.model = self.init_model(num_data, self.num_states, self.num_actions)
        # self.model

    def init_model(self, num_data, ss, aa) -> np.ndarray:
        A = 0.99
        B = 0.05
        C = 0.125
        var = 0.001
        V = 10 ** -5 * np.ones((ss, aa, ss))
        ra = np.ones(aa)
        y = np.ones(num_data)
        a = np.ones(num_data + 1)
        y[0] = 5.5
        for t in range(2, num_data - 1):
            a[t] = np.random.choice([a for a in range(aa)], 1)[0] + 1
            y[t] = A * y[t - 1] + B * a[t] - C + B * np.random.normal(0, 1, 1)

        yy_len = (np.max(y) - np.min(y)) / ss
        bins = np.linspace(np.min(y), np.max(y), ss)
        yy = np.digitize(y, bins)

        for t in range(1, num_data - 1):
            V[yy[t].astype(np.int64) - 1, a[t].astype(np.int64) - 1, yy[t - 1].astype(np.int64) - 1] = V[yy[t].astype(
                np.int64) - 1, a[t].astype(np.int64) - 1, yy[t - 1].astype(np.int64) - 1] + 1

        for at in range(aa):
            for s1 in range(ss):
                V[:, at, s1] = V[:, at, s1] / np.sum(V[:, at, s1])
        return V

    def generate_state(self, action, stop_action, state, stop_state, past_history) -> np.int:
        # TODO: Create some model
        obs_state = np.zeros((2))
        if stop_action == 1 and stop_state == 1:
            obs_state[0] = np.random.choice(np.arange(self.num_states), 1, p=self.model[:, action, state])[0]
            #obs_state[1] = np.random.choice([0, 1], 1, p=[0.05, 0.95])[0]
            obs_state[1] = 1
        else:
            obs_state[0] = state
            obs_state[1] = 0
        return obs_state.astype(int)
