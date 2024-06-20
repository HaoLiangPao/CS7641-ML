import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
import json5

# Load configuration
with open("configs.json5") as f:
    config = json5.load(f)

MAX_ATTEMPTS = config["shared"]["max_attempts"]
MAX_ITERS = config["shared"]["max_iters"]

# ===== 1. Define optimization problems =====

# Simple Version
# Four Peaks Problem (Maximization, favors SA)
four_peaks_fitness = mlrose.FourPeaks(t_pct=0.15)
four_peaks_simple_problem = mlrose.DiscreteOpt(
    length=20,
    fitness_fn=four_peaks_fitness,
    maximize=True,
    max_val=2
)

# # Custom Knapsack Problem (Minimization, favors GA)
# class CustomKnapsack(mlrose.Knapsack):
#     def evaluate(self, state):
#         total_weight = np.dot(state, self.weights)
#         total_value = np.dot(state, self.values)
#         if total_weight > self.max_weight:
#             return 0
#         # Modify the fitness function to minimize the difference between total weight and a target weight
#         target_weight = self.max_weight * 0.9
#         fitness = abs(total_weight - target_weight)
#         return fitness

# weights = [10, 5, 2, 8, 15, 7, 3, 20, 6, 1]
# values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# knapsack_fitness = CustomKnapsack(weights, values, max_weight_pct=0.6)
# knapsack_simple_problem = mlrose.DiscreteOpt(
#     length=10,
#     fitness_fn=knapsack_fitness,
#     maximize=True,
#     max_val=2
# )

# ===== 2. Define and Run the Runner =====

# Define runner for Four Peaks Problem
def run_four_peaks(problem):
    runner = mlrose.runners.GARunner(
        problem=problem,
        experiment_name="four_peaks",
        output_directory="./four_peaks_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=MAX_ATTEMPTS,
        population_sizes=[200],
        mutation_rates=[0.1]
    )
    return runner.run()

# Define runner for Knapsack Problem
def run_knapsack(problem):
    runner = mlrose.runners.GARunner(
        problem=problem,
        experiment_name="knapsack",
        output_directory="./knapsack_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=MAX_ATTEMPTS,
        population_sizes=[200],
        mutation_rates=[0.1]
    )
    return runner.run()

# Run experiments
four_peaks_results = run_four_peaks(four_peaks_simple_problem)
# knapsack_results = run_knapsack(knapsack_simple_problem)

# ===== 3. Plot the results =====

def plot_results(results, problem_name):
    aggregated_curves = results[0]
    detailed_curves = results[1]

    plt.figure(figsize=(14, 6))

    for stats in detailed_curves:
        iteration = stats[1]
        time = stats[2]
        fitness = stats[3]
        fevals = stats[4]
        population_size = stats[5]
        mutation_rate = stats[6]
        max_iters = stats[7]

    plt.plot(
        [stats[4] for stats in detailed_curves],
        [stats[3] for stats in detailed_curves],
        label=label,
    )

    plt.title(f"Performance Comparison for {problem_name}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

plot_results(four_peaks_results, "Four Peaks (simple)")
# plot_results(knapsack_results, "Knapsack (simple)")
