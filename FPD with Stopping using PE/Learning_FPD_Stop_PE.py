import numpy as np
from matplotlib import pyplot as plt

# from FPD_Stop_PE import FPD_Stop_PE
from Entities import Agent, Environment


class Experiment:

    def __init__(self, num_states: int, num_actions: int,
                 num_steps: int, num_mc_runs: int,
                 horizon: np.int, ideal_s: np.ndarray, ideal_a: np.ndarray, w: np.float, mu: np.float) -> None:
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
        # If we know the model of the Environment in advance e.g. we did population research
        self.known_model = 1
        self.d_storage = np.zeros((self.horizon, self.num_actions, 2, self.num_states, 2, num_mc_runs))
        self.h_storage = np.zeros((self.horizon, self.num_states, 2, num_mc_runs))
        h_init = 1
        self.h_storage[-1] = h_init
        self.state_sequence = np.zeros((num_mc_runs, horizon))
        self.selected_state_sequence = np.zeros((num_mc_runs, horizon))
        self.stop_action_sequence = np.ones((num_mc_runs, horizon))
        # self.experiment = FPD_Stop_PE(self.num_states, self.num_actions, self.horizon, self.ideal_s, self.ideal_a,
        #                              self.w, self.mu)

    def run(self) -> None:
        """
        Run experiment and print errors.
        """
        # self.experiment.evaluate_FPD()
        # Calculating and saving optimal_policy
        # self.plot_opt_policy(self.num_steps - 10, self.num_steps, opt_policy)
        for i in range(self.num_mc_runs):
            mcs = MonteCarloSimulation(self.num_states, self.num_actions, self.num_steps, self.ideal_s, self.ideal_a,
                                       self.w, self.mu, self.known_model, self.horizon)
            mcs.perform_single_run()
            mcs.print_errors()
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

    def __init__(self, num_states, num_actions, num_steps, ideal_s, ideal_a, w, mu, known_model, horizon) -> None:
        """
        Initialize a single Monte Carlo simulation.
        """
        self.num_states, self.num_actions = num_states, num_actions
        self.num_steps = num_steps
        self.known_model = known_model
        self.history = np.zeros((2, 2), dtype=int)
        seed = 2
        self.ideal_s, self.ideal_a = ideal_s, ideal_a
        self.w, self.mu = w, mu
        self.horizon = horizon
        self.agent = Agent(self.num_states, self.num_actions, self.horizon, self.ideal_s, self.ideal_a, self.w, self.mu)
        self.system = Environment(self.num_states, self.num_actions)
        self.errors = np.zeros((horizon, 1))
        self.predicted_states = np.zeros((horizon + 1, 2), dtype=int)
        self.states = np.zeros((horizon + 1, 2), dtype=int)
        self.init_history()


    def init_history(self):
        """
        Initialize history of the single run simulation
        :return:
        """
        # history[0,:] = [action, stop_action]
        # history[1,:] = [state, stop_state]
        history = np.zeros((2, 2), dtype=int)
        self.history = self.agent.history
        #return history

    def perform_single_run(self):
        """
        Perform a single run for the given number of decision steps and return some error plots.
        """
        stop_action = 1
        stop_state = 1
        prev_stop = 1
        action = 0
        for step in range(self.horizon):
            # estimation phase
            pred_state = self.agent.predict_state(self.history)
            self.predicted_states[step, :] = pred_state
            current_state = self.system.generate_state(self.history[0, 0], self.history[0, 1], self.history[1, 0],
                                                       self.history[1, 1], 0)
            self.states[step, :] = current_state
            self.history[1, :] = current_state
            # store errors
            self.errors[step] = np.abs(pred_state[0] - current_state[0])
            self.agent.update_V(current_state[0], self.history[0, 0], self.history[1, 0])
            self.history[0, :] = self.agent.generate_action(current_state[0], step)
            self.agent.history = self.history
            if self.history[1, 1] == 0:
                self.agent.stop()
            self.agent.estimate_model()
            if (step % self.num_steps == (self.num_steps - 1)) and self.agent.continues == 1:
                #self.agent.estimate_model()
                self.agent.evaluate_FPD()
            self.agent.t = self.horizon - 1
            #if self.agent.continues == 0:
                #self.fill_rest()

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
        print("Predicted states: ")
        print(self.predicted_states)
        print("Observed states: ")
        print(self.states)

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
