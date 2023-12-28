import numpy as np
from matplotlib import pyplot as plt
import time
from prettytable import PrettyTable

# from FPD_Stop_PE import FPD_Stop_PE
from Entities import Agent, Environment


class Experiment:

    def __init__(self, num_states: int, num_actions: int,
                 num_steps: int, num_mc_runs: int,
                 horizon: np.int, ideal_s: np.ndarray, ideal_a: np.ndarray, w: np.float, mu: np.float,
                 len_sim: np.int, q: np.float) -> None:
        """
        Initialize experiment.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.num_mc_runs = num_mc_runs
        self.horizon = horizon
        self.ideal_s = ideal_s
        self.ideal_a = ideal_a
        self.w = w
        self.mu = mu
        self.len_sim = len_sim
        # If we know the model of the Environment in advance e.g. we did population research
        self.known_model = 1
        self.d_storage = np.zeros((self.horizon, self.num_actions, self.num_states, num_mc_runs))
        self.h_storage = np.zeros((self.horizon, self.num_states, num_mc_runs))
        h_init = 1
        self.h_storage[-1] = h_init
        self.state_sequence = np.zeros((num_mc_runs, 2, len_sim))
        self.action_sequence = np.ones((num_mc_runs, 2, len_sim))
        self.stop_time_sequence = np.ones((num_mc_runs, 2, len_sim))
        self.time_cost = np.zeros((num_mc_runs, 2))
        self.q = q

    def run(self) -> None:
        """
        Run experiment and print errors.
        """
        # self.experiment.evaluate_FPD()
        # Calculating and saving optimal_policy
        # self.plot_opt_policy(self.num_steps - 10, self.num_steps, opt_policy)
        for i in range(self.num_mc_runs):
            t0 = time.time()
            mcs = MonteCarloSimulation(self.num_states, self.num_actions, self.num_steps, self.ideal_s, self.ideal_a,
                                       self.w, self.mu, self.known_model, self.horizon, self.q, self.len_sim)
            mcs.perform_single_run()
            t1 = time.time()
            # print("time single run:", t1-t0)
            self.load_results(mcs, i, 0, t1 - t0)
            # mcs.print_errors()

            # mcs2 = MonteCarloSimulation(self.num_states, self.num_actions, self.num_steps, self.ideal_s, self.ideal_a,
            #                           self.w, self.mu, self.known_model, self.horizon)
            # mcs2.perform_free_run()
            t2 = time.time()
            mcs3 = MonteCarloSimulation(self.num_states, self.num_actions, self.num_steps, self.ideal_s, self.ideal_a,
                                        self.w, self.mu, self.known_model, self.horizon, self.q, self.len_sim)
            mcs3.perform_single_dynamic_run()
            t3 = time.time()
            # print("time dynamic run:", t3-t2)
            self.load_results(mcs3, i, 1, t3 - t2)
            # errors = mcs.error_compute()
            # print(errors)
        # print("Median value of state in classic model was: ")
        # print(np.median(np.median(self.state_sequence[:, 0, :], axis=0), axis=0))
        # print("Median value of action in classic model was: ")
        # print(np.median(np.median(self.action_sequence[:, 0, :], axis=0), axis=0))
        # print("Median value of state in dynamic model was: ")
        # print(np.median(np.median(self.state_sequence[:, 1, :], axis=0), axis=0))
        # print("Median value of action in dynamic model was: ")
        # print(np.median(np.median(self.action_sequence[:, 1, :], axis=0), axis=0))
        # print("Median value of stop_time was: ")
        # print(np.median(self.stop_time_sequence[:, 1, :], axis=1))
        # print("Observed states: ")
        # print(self.states)
        # bin_edges = np.arange(-0.5, self.horizon + 0.5, 1)
        # plt.hist(np.median(self.stop_time_sequence[:, 1, :], axis=0), bins=bin_edges)
        # plt.xticks(np.arange(0, self.horizon + 1, 1))
        # plt.title("Controlled dynamic run")
        # plt.xlabel("time")
        # plt.show()
        #
        # bin_edges = np.arange(-0.5, self.num_states + 0.5, 1)
        # plt.hist(np.median(self.state_sequence[:, 1, :], axis=0), bins=bin_edges)
        # plt.xticks(np.arange(0, self.num_states + 1, 1))
        # plt.title("Controlled dynamic run")
        # plt.xlabel("States")
        # plt.show()
        #
        # bin_edges = np.arange(-0.5, self.num_states + 0.5, 1)
        # plt.hist(np.median(self.state_sequence[:, 0, :], axis=0), bins=bin_edges)
        # plt.xticks(np.arange(0, self.num_states + 1, 1))
        # plt.title("Controlled fixed horizon run")
        # plt.xlabel("States")
        # plt.show()
        #
        # bin_edges = np.arange(-0.5, self.num_actions + 0.5, 1)
        # plt.hist(np.median(self.action_sequence[:, 1, :], axis=0), bins=bin_edges)
        # plt.xticks(np.arange(0, self.num_actions + 1, 1))
        # plt.title("Controlled dynamic run")
        # plt.xlabel("Actions")
        # plt.show()
        #
        # bin_edges = np.arange(-0.5, self.num_actions + 0.5, 1)
        # plt.hist(np.median(self.action_sequence[:, 0, :], axis=0), bins=bin_edges)
        # plt.xticks(np.arange(0, self.num_actions + 1, 1))
        # plt.title("Controlled fixed horizon run")
        # plt.xlabel("Actions")
        # plt.show()

        self.save_results(mcs3)
        self.save_images(mcs3)

    def save_results(self, mcs):

        # Specify the Column Names while initializing the Table
        myTable = PrettyTable(["RUNS", "Median", "Mean", "STD DER", "MAX", "MIN"])
        # Add rows
        myTable.add_row(["Regular ", np.median(self.time_cost[:, 0], axis=0),
                         np.mean(self.time_cost[:, 0], axis=0),
                         np.std(self.time_cost[:, 0], axis=0),
                         np.max(self.time_cost[:, 0], axis=0),
                         np.min(self.time_cost[:, 0], axis=0)])
        myTable.add_row(["Dynamic", np.median(self.time_cost[:, 1], axis=0),
                         np.mean(self.time_cost[:, 1], axis=0),
                         np.std(self.time_cost[:, 1], axis=0),
                         np.max(self.time_cost[:, 1], axis=0),
                         np.min(self.time_cost[:, 1], axis=0)])
        filename = "Inputs_" + str(self.len_sim) + "_" + str(self.horizon) + "_q_" + str(self.q) + "_mu_" + str(
            self.mu) + "_w_" + str(self.w) + "_s_" + str(self.ideal_s) + "_a_" + str(self.ideal_a) + ".txt"
        with open(filename, 'w') as w:
            w.write(myTable.get_string())
        print(myTable)

    def save_images(self, mcs):
        bin_edges = np.arange(-0.5, self.horizon + 0.5, 1)
        plt.hist(np.median(self.stop_time_sequence[:, 1, :], axis=0), bins=bin_edges)
        plt.xticks(np.arange(0, self.horizon + 1, 1))
        plt.title("Controlled dynamic run")
        plt.xlabel("time")
        plt.grid()
        # plt.show()
        filename = "plot_stop_time_" + str(self.len_sim) + "_" + str(self.horizon) + "_q_" + str(self.q) + "_mu_" + str(
            self.mu) + "_w_" + str(self.w) + ".png"
        plt.savefig(filename)
        plt.close()

        bin_edges = np.arange(-0.5, self.num_states + 0.5, 1)
        plt.hist(np.median(self.state_sequence[:, 1, :], axis=0), bins=bin_edges)
        plt.xticks(np.arange(0, self.num_states + 1, 1))
        plt.title("Controlled dynamic run")
        plt.xlabel("States")
        plt.grid()
        # plt.show()
        filename = "plot_states_dynamic_" + str(self.len_sim) + "_" + str(self.horizon) + "_q_" + str(
            self.q) + "_mu_" + str(
            self.mu) + "_w_" + str(self.w) + ".png"
        plt.savefig(filename)
        plt.close()

        bin_edges = np.arange(-0.5, self.num_states + 0.5, 1)
        plt.hist(np.median(self.state_sequence[:, 0, :], axis=0), bins=bin_edges)
        plt.xticks(np.arange(0, self.num_states + 1, 1))
        plt.title("Controlled fixed horizon run")
        plt.xlabel("States")
        plt.grid()
        # plt.show()
        filename = "plot_states_regular_" + str(self.len_sim) + "_" + str(self.horizon) + "_q_" + str(
            self.q) + "_mu_" + str(
            self.mu) + "_w_" + str(self.w) + ".png"
        plt.savefig(filename)
        plt.close()

        bin_edges = np.arange(-0.5, self.num_actions + 0.5, 1)
        plt.hist(np.median(self.action_sequence[:, 1, :], axis=0), bins=bin_edges)
        plt.xticks(np.arange(0, self.num_actions + 1, 1))
        plt.title("Controlled dynamic run")
        plt.xlabel("Actions")
        plt.grid()
        # plt.show()
        filename = "plot_actions_dynamic_" + str(self.len_sim) + "_" + str(self.horizon) + "_q_" + str(
            self.q) + "_mu_" + str(
            self.mu) + "_w_" + str(self.w) + ".png"
        plt.savefig(filename)
        plt.close()

        bin_edges = np.arange(-0.5, self.num_actions + 0.5, 1)
        plt.hist(np.median(self.action_sequence[:, 0, :], axis=0), bins=bin_edges)
        plt.xticks(np.arange(0, self.num_actions + 1, 1))
        plt.title("Controlled fixed horizon run")
        plt.xlabel("Actions")
        plt.grid()
        # plt.show()
        filename = "plot_actions_regular_" + str(self.len_sim) + "_" + str(self.horizon) + "_q_" + str(
            self.q) + "_mu_" + str(
            self.mu) + "_w_" + str(self.w) + ".png"
        plt.savefig(filename)
        plt.close()

    def load_results(self, mcs, index, order, time_cost):
        """

        :param mcs:
        :param index: number of run from which the results are loaded
        :param order: 0 for normal run, 1 for dynamic
        :return:
        """
        self.state_sequence[index, order, :] = mcs.states
        self.action_sequence[index, order, :] = mcs.actions
        self.stop_time_sequence[index, order, :] = mcs.stopping_time
        self.time_cost[index, order] = time_cost

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

    def __init__(self, num_states, num_actions, num_steps, ideal_s, ideal_a, w, mu, known_model, horizon, q,
                 len_sim=500) -> None:
        """
        Initialize a single Monte Carlo simulation.
        """
        self.num_states, self.num_actions = num_states, num_actions
        self.num_steps = num_steps
        self.known_model = known_model
        self.history = np.zeros(2, dtype=int)
        seed = 2
        self.ideal_s, self.ideal_a = ideal_s, ideal_a
        self.w, self.mu = w, mu
        self.horizon = horizon
        self.len_sim = len_sim
        self.agent = Agent(self.num_states, self.num_actions, self.horizon, self.ideal_s, self.ideal_a, self.w, self.mu,
                           q)
        self.system = Environment(self.num_states, self.num_actions)
        self.errors = np.zeros((len_sim, 1))
        self.predicted_states = np.zeros(len_sim, dtype=int)
        self.states = np.zeros(len_sim, dtype=int)
        self.actions = np.zeros(len_sim, dtype=int)
        self.stopping_time = np.zeros(len_sim, dtype=int)
        self.init_history()

    def init_history(self):
        """
        Initialize history of the single run simulation
        :return:
        """
        # history[0,:] = [action, stop_action]
        # history[1,:] = [state, stop_state]
        history = np.zeros(2, dtype=int)
        self.history = self.agent.history
        # return history

    def perform_free_run(self):
        """
        Perform a single run without learning and any preference
        :return:
        """
        for step in range(self.horizon):
            # estimation phase
            current_state = self.system.generate_state(self.history[0], self.history[1], 0, 0)
            self.states[step] = current_state
            self.history[1] = current_state
            # store errors
            self.history[0] = np.random.choice([a for a in range(self.num_actions)], 1)[0]

        # bin_edges = np.arange(-0.5, 10.5, 1)
        # plt.hist(self.states, bins=bin_edges)
        # plt.xticks(np.arange(0, 11, 1))
        # plt.title("Free_run")
        # plt.show()

    def perform_single_run(self):
        """
        Perform a single run for the given number of decision steps and return some error plots.
        """
        for step in range(self.len_sim):
            # estimation phase
            current_state = self.system.generate_state(self.history[0], self.history[1], 0, 0)
            self.states[step] = current_state
            self.history[1] = current_state
            # store errors
            self.agent.update_V(current_state, self.history[0], self.history[1])
            self.agent.estimate_model()
            self.agent.t = self.horizon - 1
            self.agent.h[:, :] = 1
            for i in range(self.horizon):
                self.agent.evaluate_PE_short()
                self.agent.evaluate_r_o()
                self.agent.normalize_r_o()
                action = self.agent.generate_action(current_state)
            # We end DM cycle if horizon time is reached or if we stopped DM before
            self.history[0] = action
            self.actions[step] = action
            self.agent.history = self.history

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
        # print(self.agent.stop_time)
        # print("Predicted states: ")
        # print(self.predicted_states)
        # print("Observed states: ")
        # print(self.states)
        # bin_edges = np.arange(-0.5, 10.5, 1)
        # plt.hist(self.states, bins=bin_edges)
        # plt.xticks(np.arange(0, 11, 1))
        # plt.title("Controlled run")
        # plt.show()

    def perform_single_dynamic_run(self):
        """
        Perform a single run for the given number of decision steps and return some error plots.
        """
        stop_action = 1
        stop_state = 1
        prev_stop = 1
        action = 0
        for step in range(self.len_sim):
            # estimation phase
            current_state = self.system.generate_state(self.history[0], self.history[1], 0, 0)
            self.states[step] = current_state
            self.history[1] = current_state
            # store errors
            self.agent.update_V(current_state, self.history[0], self.history[1])
            self.agent.estimate_model()
            self.agent.t = self.horizon - 1
            self.agent.h[:, :] = 1
            self.agent.h_long[:, :, :] = 1
            while self.agent.continues > 0:
                self.agent.evaluate_PE()
                self.agent.evaluate_FPD_Stop()
                action = self.agent.stop_decision()
            self.agent.continues = 1
            # We end DM cycle if horizon time is reached or if we stopped DM before
            self.history[0] = action
            self.agent.d_io_long.fill(0)
            self.actions[step] = action
            self.stopping_time[step] = self.agent.stop_time
            self.agent.history = self.history

        # print("Observed states: ")
        # print(self.states)
        # bin_edges = np.arange(-0.5, 10.5, 1)
        # plt.hist(self.states, bins=bin_edges)
        # plt.xticks(np.arange(0, 11, 1))
        # plt.title("Controlled dynamic run")
        # plt.show()

    def perform_single_run_floating_horizon(self, floating_horizon: np.int, init: bool = False):
        stop_action = 1
        stop_state = 1
        prev_stop = 1
        action = 0
        for step in range(self.horizon):
            # estimation phase
            pred_state = self.agent.predict_state(self.history)
            self.predicted_states[step] = pred_state
            current_state = self.system.generate_state(self.history[0], self.history[1], 0, 0)
            self.states[step] = current_state
            self.history[1] = current_state
            # store errors
            self.errors[step] = np.abs(pred_state - current_state)
            self.agent.update_V(current_state, self.history[0], self.history[1])
            self.history[0] = self.agent.generate_action(current_state, step)

            self.agent.history = self.history
            if self.agent.continues == 1:
                self.agent.estimate_model()
                self.agent.evaluate_FPD_floating_horizon(floating_horizon, init)
        # self.agent.t = self.horizon - 1

    def print_errors(self):
        print(np.sum(self.errors))

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


class BaseAgent:

    def __init__(self, num_states: int, num_actions: int):
        """
        Initialize agent class without information fusion.
        """
        self.num_states, self.num_actions = num_states, num_actions
        self.occurrence_table = self.initialize_prior_occurrence_table()
        self.stop_ind = 1
        self.ro = np.ones((num_states, 2)) / (num_states * 2)
        self.history = np.zeros((2, 1))

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
