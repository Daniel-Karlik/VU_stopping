import numpy as np
from matplotlib import pyplot as plt

from FPD_Stop_PE import FPD_Stop_PE


class Experiment:

    def __init__(self, num_states: int, num_actions: int,
                 steps: int, num_mc_runs: int, model_i: np.ndarray, model: np.ndarray,
                 horizon: np.ndarray, ideal_s: np.ndarray, ideal_a: np.ndarray, w: np.float, mu: np.float) -> None:
        """
        Initialize experiment.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.steps = steps
        self.num_mc_runs = num_mc_runs
        self.model = model
        self.horizon = horizon
        self.ideal_s = ideal_s
        self.ideal_a = ideal_a
        self.w = w
        self.mu = mu
        # If we know the model of the Environment in advance e.g. we did population research
        self.known_model = 1
        self.d_storage = np.zeros((self.num_actions, 2, self.num_states, 2, self.horizon))
        self.h_storage = np.zeros((self.num_states, 2, self.horizon))
        self.ro = np.ones((num_actions, 2, num_states, 2, horizon))
        h_init = 1
        self.h_storage[-1] = h_init
        self.state_sequence = np.zeros((self.horizon, num_mc_runs))
        self.selected_state_sequence = np.zeros((self.horizon, num_mc_runs))
        self.stop_action_sequence = np.zeros((self.horizon, num_mc_runs))
        self.system = Environment()

    def run(self) -> None:
        """
        Run experiment and print errors.
        """
        # init phase of the experiment

        # Calculating and saving optimal_policy
        # self.plot_opt_policy(self.num_steps - 10, self.num_steps, opt_policy)
        for i in range(self.num_mc_runs):
            mcs = MonteCarloSimulation(self.num_states, self.num_actions, self.num_steps, self.model_i, self.model,
                                       self.q_i, self.experiment, self.known_model, self.system)
            self.state_sequence[i] = mcs.perform_single_run(self.stop_action_sequence[i],
                                                            self.selected_state_sequence[i])
            # errors = mcs.error_compute()
            # print(errors)

    def plot_results(self, results) -> None:
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
                 q_i: np.ndarray, policy: np.ndarray, known_model: int, system) -> None:
        """
        Initialize a single Monte Carlo simulation.
        """
        self.num_states, self.num_actions = num_states, num_actions
        self.known_model = known_model
        self.history = np.zeros((num_steps + 1, 2), dtype=int)
        self.prediction_table = np.zeros((num_steps + 1, 2), dtype=int)
        self.model_i = model_i
        self.model = model
        self.q_i = q_i
        seed = 2
        self.system = system
        self.agent = BaseAgent(self.num_states, self.num_actions, self.model_i, self.model, self.q_i, seed)
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
