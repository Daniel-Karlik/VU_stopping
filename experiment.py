from pathlib import Path
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from Environment import Environment, BaseAgent
from ParametricEstimator import ParametricEstimator
# Simulation settings of the experiments
SEED = 1


class Experiment:

    def __init__(self, num_steps: int, num_mc_runs: int) -> None:
        """
        Initialize experiment.
        """
        self.num_steps = num_steps
        self.num_mc_runs = num_mc_runs

    def _get_averaged_kld(self) -> np.ndarray:
        """
        Return an average of Kullback-Leibler divergence over the number of Monte Carlo runs.
        """
        avg_kld = np.zeros((1, 3), dtype=float)
        for i in range(self.num_mc_runs):
            avg_kld += MonteCarloSimulation(self.num_steps).perform_single_run()
        return avg_kld / self.num_mc_runs

    def run(self) -> None:
        """
        Run experiment.
        """
        results = {}
        self.plot_results(results)

    def plot_results(self, results: Dict[float, np.ndarray]) -> None:
        """
        Plot figures.
        """
        fontsize = 16
        ext_agent_kld = []
        base_agent_kld = []
        agent_kld = []

        for w in self.w_span:
            ext_agent_kld.append(results[w][0][0])
            base_agent_kld.append(results[w][0][1])
            agent_kld.append(results[w][0][2])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.w_span, base_agent_kld, c='b', ls='--', linewidth=4)
        ax.plot(self.w_span, agent_kld, c='r', ls='-', linewidth=4)
        plt.xlabel(r'$\mathsf{w}$', fontsize=fontsize)
        plt.ylabel('KLD', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        #plt.savefig(Path(__file__).parent / f'figures/results_for_alpha_{self.alpha.__str__().replace(".", "-")}.png')
        plt.show()


class MonteCarloSimulation:

    def __init__(self, num_steps: int) -> None:
        """
        Initialize a single Monte Carlo simulation.
        """
        self.num_steps = num_steps
        self.environment = Environment()
        self.agent = BaseAgent()

    def perform_single_run(self) -> np.ndarray:
        """
        Perform a single run for the given number of decision steps and return a final Kullback-Leibler divergence.
        """
        for step in range(self.num_steps):
            action = self.environment.action()
            observable_state = self.environment.state()
            self.agent.update_occurrence_table(action, observable_state)
            self.agent.update_occurrence_table(action, observable_state)
        return self._compute_kld()

    def _compute_kld(self) -> np.ndarray:
        """
        Compute Kullback_Leibler divergences.
        """
        kld = np.zeros((self.agent.num_actions, 3), dtype=float)
        for action in range(self.agent.num_actions):
            env_prob = self._to_probabilities(self.environment.occurrence_table[action])
            kld[action][2] = sum(np.multiply(env_prob, np.divide(env_prob, self._to_probabilities(self.agent.occurrence_table[action]))))
        return kld

    @staticmethod
    def _to_probabilities(occurrences: np.ndarray) -> np.ndarray:
        """
        Normalize occurrence table $V_{o_{\M{a}}}$ to probabilities.
        """
        return occurrences / sum(occurrences)
