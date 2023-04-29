from pathlib import Path
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from Environment import Environment, BaseAgent
from ParametricEstimator import ParametricEstimator
# Simulation settings of the experiments
SEED = 1


class Experiment:

    def __init__(self, num_states: int, num_actions: int,
                 num_steps: int, num_mc_runs: int, model_i: np.ndarray, model: np.ndarray,
                 q_i: np.ndarray) -> None:
        """
        Initialize experiment.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.num_mc_runs = num_mc_runs
        self.model_i = model_i
        self.model = model
        self.q_i = q_i
        self.agent = BaseAgent(self.num_states, self.num_actions, self.model_i, self.model,
                               self.q_i, seed=2)
        self.d_storage = np.zeros((self.num_steps, self.num_actions, self.agent.num_stop_actions, self.num_states,
                                   self.agent.num_stop_states))
        self.h_storage = np.zeros((self.num_steps, self.num_states, self.agent.num_stop_states))
        h_init = 1
        self.h_storage[-1] = h_init
        self.state_sequence = np.zeros((num_mc_runs, num_steps))
        self.selected_state_sequence = np.zeros((num_mc_runs, num_steps))
        self.stop_action_sequence = np.ones((num_mc_runs, num_steps))

    def _get_averaged_kld(self) -> np.ndarray:
        """
        Return an average of Kullback-Leibler divergence over the number of Monte Carlo runs.
        """
        pass

    def run(self) -> None:
        """
        Run experiment and print errors.
        """
        self.calculate_h_d_fun()
        # Calculating and saving optimal_policy
        opt_policy = self.optimal_policy()
        self.plot_opt_policy(self.num_steps-10, self.num_steps, opt_policy)
        for i in range(self.num_mc_runs):
            mcs = MonteCarloSimulation(self.num_states, self.num_actions, self.num_steps, self.model_i, self.model,
                                       self.q_i, opt_policy)
            self.state_sequence[i] = mcs.perform_single_run(self.stop_action_sequence[i], self.selected_state_sequence[i])
            #errors = mcs.error_compute()
            #print(errors)

    def h_fun(self, time: int, state: int, stop_state: int) -> float:
        """
        Calculates h function values based on previous d function values and ideal decision rule
        :param time:
        :param state:
        :param stop_state:
        :param self.q_i:
        :param self.d_storage:
        :return:
        """
        output = 0
        for i in range(self.num_actions):
            for stop in range(self.agent.num_stop_actions):
                output += self.agent.ideal_decision_rule(i, stop, state, stop_state) * \
                          np.exp(-self.d_storage[time+1, i, stop, state, stop_state])
        return output

    def d_fun(self, time: int, stop_action: int, prev_state: int) -> float:
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
        num_stop_states = 2
        for i in range(self.num_states):
            for stop in range(self.agent.num_stop_states):
                model_reg = self.agent.model_prob(i, stop_action, stop, prev_state)
                id_mod = self.agent.model_ideal_prob(i, stop, stop_action, prev_state)
                # print(id_mod)
                if model_reg == 0:
                    output += 0
                else:
                    output += model_reg*np.log(model_reg/(id_mod*self.h_storage[time, i, stop]))
        fun_output = output
        return fun_output

    def calculate_h_d_fun(self) -> None:
        """
        Fills up self.h_storage and self.d_storage with proper values, using backward recursion starting with
        self.h_storage[-1] = h_init and known values of self.model_i, self.model, self.q_i
        :return:
        self.h_storage and self.d_storage filled with values for finding the optimal policy
        """
        t_init = self.num_steps - 1
        for t_ind in range(t_init, -1, -1):
            if t_ind != t_init:
                for state in range(self.num_states):
                    for stop_state in range(self.agent.num_stop_states):
                        self.h_storage[t_ind, state, stop_state] = self.h_fun(t_ind, state, stop_state)
            # range(start, stop, step)
            for action in range(self.num_actions):
                for stop_action in range(self.agent.num_stop_actions):
                    for state in range(self.num_states):
                        for stop_state in range(self.agent.num_stop_states):
                            self.d_storage[t_ind, action, stop_action, state, stop_state] = \
                                self.d_fun(t_ind, stop_action, state)

    def normalize_d_fun(self) -> float:
        pass

    def optimal_policy(self) -> np.ndarray:
        opt_policy = np.zeros((self.num_steps, self.num_actions, self.agent.num_stop_actions, self.num_states,
                               self.agent.num_stop_states))
        for t in range(self.num_steps-1, -1, -1):
            for action in range(self.num_actions):
                for stop_action in range(self.agent.num_stop_actions):
                    for state in range(self.num_states):
                        for stop_state in range(self.agent.num_stop_states):
                            opt_policy[t, action, stop_action, state, stop_state] = \
                                self.agent.ideal_decision_rule(action, stop_action, state, stop_state) * \
                                (np.exp(-self.d_storage[t, action, stop_action, state, stop_state]) /
                                 self.h_storage[t, state, stop_state])
        # print(np.shape(opt_policy))
        return opt_policy

    def plot_results(self, results: Dict[float, np.ndarray]) -> None:
        """
        Plot figures.
        """
        pass

    def plot_opt_policy(self, low: int, high: int, policy: np.ndarray):
        """
        Plots calculated optimal policy
        :param low:
        :param high:
        :param policy:
        :return:
        """
        for state in range(self.num_states):
            plt.plot(policy[low:high, 0, 0, state, 1], label=str(state))
        plt.legend(loc="lower left")
        plt.show()


class MonteCarloSimulation:

    def __init__(self, num_states: int, num_actions: int, num_steps: int, model_i: np.ndarray, model: np.ndarray,
                 q_i: np.ndarray, policy: np.ndarray) -> None:
        """
        Initialize a single Monte Carlo simulation.
        """
        self.num_states, self.num_actions = num_states, num_actions
        self.num_steps = num_steps
        self.environment = Environment(self.num_states, self.num_actions, seed=2)
        self.history = np.zeros((num_steps+1, 2), dtype=int)
        self.prediction_table = np.zeros((num_steps+1, 2), dtype=int)
        self.model_i = model_i
        self.model = model
        self.q_i = q_i
        self.agent = BaseAgent(self.num_states, self.num_actions, self.model_i, self.model, self.q_i, seed=2)
        self.policy = policy

    def init_history(self):
        """
        Initialize history of the single run simulation
        :return:
        """
        history = np.zeros((self.num_steps + 1, 2), dtype=int)
        history[0, 0:1] = self.agent.history[0:1]
        return history

    def perform_single_run(self, stop_action_sequence: np.ndarray, selected_state_sequence: np.ndarray) -> np.ndarray:
        """
        Perform a single run for the given number of decision steps and return some error plots.
        """
        state_sequence = np.random.choice(np.arange(0, self.num_states), p=self.model, size=self.num_steps)
        stop_action = 1
        stop_state = 1
        prev_stop = 1
        action = 0
        for step in range(self.num_steps):
            cur_stop_action = self.stopping_decision(step, action, state_sequence[step], stop_state)
            if prev_stop != cur_stop_action:
                selected_state = state_sequence[step]
            if cur_stop_action == 1:
                # process continues
                cur_state = state_sequence[step]
            else:
                cur_state = selected_state
            stop_action_sequence[step] = cur_stop_action
            selected_state_sequence[step] = cur_state
            prev_stop = cur_stop_action
            stop_state = cur_stop_action
            # We make prediction before observing new state
            # predicted_state = self.agent.make_prediction()
            # # The Environment generates new state
            # observable_state = self.environment.generate_state()
            # self.agent.update_occurrence_table(observable_state)
            # # Agent generates action
            # # TODO maybe better way to handle history + generating new action in agent class
            # action = self.agent.generate_action()
            # # Histories of environment and agent are updated
            # self.agent.update_history(action, observable_state)
            # self.environment.update_history(action, observable_state)
            # self.store_pred(observable_state, predicted_state)
        return state_sequence


    def _compute_kld(self) -> np.ndarray:
        """
        Compute Kullback_Leibler divergences.
        """
        pass

    def error_compute(self) -> np.ndarray:
        """
        Compute number of errors from observed and predicted states
        :return: cumulative sum of errors
        """
        step_error = np.abs(self.prediction_table[:, 0] - self.prediction_table[:, 1])
        return np.sum(step_error)

    def store_pred(self, observable_state: int, predicted_state: int):
        """
        Store predicted and observed states
        :return:
        """
        self.prediction_table[self.environment.time, :] = [observable_state, predicted_state]

    def stopping_decision(self, t: int, action: int, state: int, stop_state: int) -> int:
        """
        Calculates stopping action
        :param t:
        :param action:
        :param state:
        :param stop_state:
        :param policy:
        :return:
        """
        p_stop = self.policy[t, action, 0, state, stop_state]
        # print(p_stop)
        stop_action = np.random.choice(np.arange(0, 2), p=[p_stop, 1 - p_stop])
        return stop_action

    @staticmethod
    def _to_probabilities(occurrences: np.ndarray) -> np.ndarray:
        """
        Normalize occurrence table $V_{o_{\\M{a}}}$ to probabilities.
        """
        return occurrences / sum(occurrences)
