import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CustomKnapsack(mlrose.Knapsack):
    def __init__(self, weights, values, max_weight_pct=0.35, target_weight_pct=0.3):
        super().__init__(weights, values, max_weight_pct)
        self.target_weight = np.ceil(np.sum(self.weights) * target_weight_pct)

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation. Must be the same length as the weights
            and values arrays.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        if len(state) != len(self.weights):
            raise Exception(
                """The state array must be the same size as the"""
                + """ weight and values arrays."""
            )

        # Calculate total weight and value of knapsack
        total_weight = np.sum(state * self.weights)
        total_value = np.sum(state * self.values)

        # Allow for weight constraint
        if total_weight <= self._w:
            fitness = abs(total_value - self.target_weight)
        else:
            fitness = self._w
        return fitness


def plot_op_results(
    results, problem_name, attribute_col, attribute, algorithm, iteration_list
):
    file_path = f"./optimization_results/{algorithm}-{problem_name}/{algorithm}__{algorithm}-{problem_name}__run_stats_df.csv"
    algorithm_results = pd.read_csv(file_path)

    # Filter data for plotting
    attribute_col_value = algorithm_results[attribute_col].unique()

    # Plot Iteration vs Fitness

    plt.figure(figsize=(14, 12))
    plt.subplot(3, 1, 1)
    # plt.figure(figsize=(14, 6))
    for attribute_value in attribute_col_value:
        attribute_value_data = algorithm_results[
            algorithm_results[attribute_col] == attribute_value
        ][: len(iteration_list)]
        plt.plot(
            attribute_value_data["Iteration"],
            attribute_value_data["Fitness"],
            label=f"{attribute} {attribute_value}",
        )
    plt.title(f"Iteration vs Fitness for {algorithm.upper()} ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    # plt.show()

    # Plot Iteration vs Time
    # plt.figure(figsize=(14, 6))
    plt.subplot(3, 1, 2)
    for attribute_value in attribute_col_value:
        attribute_value_data = algorithm_results[
            algorithm_results[attribute_col] == attribute_value
        ][: len(iteration_list)]
        plt.plot(
            attribute_value_data["Iteration"],
            attribute_value_data["Time"],
            label=f"{attribute} {attribute_value}",
        )
    plt.title(f"Iteration vs Time for {algorithm.upper()} ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.legend()
    # plt.show()

    # Plot Iteration vs Function Evaluations (FEvals)
    # plt.figure(figsize=(14, 6))
    plt.subplot(3, 1, 3)
    for attribute_value in attribute_col_value:
        attribute_value_data = algorithm_results[
            algorithm_results[attribute_col] == attribute_value
        ][: len(iteration_list)]
        plt.plot(
            attribute_value_data["Iteration"],
            attribute_value_data["FEvals"],
            label=f"{attribute} {attribute_value}",
        )
    plt.title(
        f"Iteration vs Function Evaluations for {algorithm.upper()} ({problem_name})"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Function Evaluations")
    plt.legend()
    # plt.show()

    plt.tight_layout()
    plt.show()

