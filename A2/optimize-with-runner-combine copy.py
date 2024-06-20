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
    length=20, fitness_fn=four_peaks_fitness, maximize=True, max_val=2
)


# # Custom Knapsack Problem (Minimization, favors GA)
# class CustomKnapsack(mlrose.CustomFitness):
#     def __init__(self, weights, values, max_weight_pct, target_weight_pct):
#         self.weights = np.array(weights)
#         self.values = np.array(values)
#         self.max_weight = max_weight_pct * np.sum(weights)
#         self.target_weight = target_weight_pct * np.sum(weights)

#     def evaluate(self, state):
#         total_weight = np.dot(state, self.weights)
#         if total_weight > self.max_weight:
#             return np.inf  # Penalize infeasible solutions
#         return abs(
#             total_weight - self.target_weight
#         )  # Minimize the distance to the target weight


# weights = [10, 5, 2, 8, 15, 7, 3, 20, 6, 1]
# values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# max_weight_pct = 0.6
# target_weight_pct = 0.5

# knapsack_fitness = CustomKnapsack(weights, values, max_weight_pct, target_weight_pct)
# knapsack_simple_problem = mlrose.DiscreteOpt(
#     length=10,
#     fitness_fn=knapsack_fitness,
#     maximize=False,  # Minimize the distance to the target weight
#     max_val=2,
# )

# ===== 2. Define and Run the Runner =====


# Define runner for Randomized Hill Climbing (RHC)
def run_rhc(problem):
    runner = mlrose.runners.RHCRunner(
        problem=problem,
        experiment_name="rhc",
        output_directory="./rhc_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        restart_list=[0, 5, 10],
        max_attempts=MAX_ATTEMPTS,
    )
    return runner.run()


# Define runner for Simulated Annealing (SA)
def run_sa(problem):
    runner = mlrose.runners.SARunner(
        problem=problem,
        experiment_name="sa",
        output_directory="./sa_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=MAX_ATTEMPTS,
        temperature_list=[1, 10, 50, 100],
    )
    return runner.run()


# Define runner for Genetic Algorithm (GA)
def run_ga(problem):
    runner = mlrose.runners.GARunner(
        problem=problem,
        experiment_name="ga",
        output_directory="./ga_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=MAX_ATTEMPTS,
        population_sizes=[200],
        mutation_rates=[0.1],
    )
    return runner.run()


# Define runner for MIMIC
def run_mimic(problem):
    runner = mlrose.runners.MIMICRunner(
        problem=problem,
        experiment_name="mimic",
        output_directory="./mimic_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=MAX_ATTEMPTS,
        population_sizes=[200],
        keep_percent_list=[0.1, 0.2, 0.3],
        use_fast_mimic=True,
    )
    return runner.run()


# Four peaks
rhc_four_peaks_simple_results = run_rhc(four_peaks_simple_problem)
sa_four_peaks_simple_results = run_sa(four_peaks_simple_problem)
ga_four_peaks_simple_results = run_ga(four_peaks_simple_problem)
mimic_four_peaks_simple_results = run_mimic(four_peaks_simple_problem)

# # Knapsack
# rhc_knapsack_simple_results = run_rhc(knapsack_simple_problem)
# sa_knapsack_simple_results = run_sa(knapsack_simple_problem)
# ga_knapsack_simple_results = run_ga(knapsack_simple_problem)
# mimic_knapsack_simple_results = run_mimic(knapsack_simple_problem)

# ===== 3. Plot the results =====


def plot_results(results, problem_name, metric):
    plt.figure(figsize=(14, 6))

    for label, result in results.items():
        print(result)
        print(len(result))
        print(type(result))

        print(result.iloc[:, 2].values)
        print(len(result.iloc[:, 2].values))
        print(type(result.iloc[:, 2].values))

        print(label)

        iteration =       result.iloc[:, 0].values
        fitness =         result.iloc[:, 1].values
        fevals =          result.iloc[:, 2].values
        time =            result.iloc[:, 3].values
        population_size = result.iloc[:, 4].values
        mutation_rate =   result.iloc[:, 5].values
        max_iters =       result.iloc[:, 6].values

        if metric == "time":
            plt.plot(time, fitness, label=label)
        elif metric == "fevals":
            plt.plot(fevals, fitness, label=label)

    plt.title(f"{metric.capitalize()} vs Fitness for {problem_name}")
    plt.xlabel(metric.capitalize())
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()


# Combine results
combined_four_peaks_simple_results = {
    "RHC": rhc_four_peaks_simple_results[0],
    # "SA": sa_four_peaks_simple_results[0],
    # "GA": ga_four_peaks_simple_results[0],
    # "MIMIC": mimic_four_peaks_simple_results[0],
}

# Plot Fitness vs Time
plot_results(combined_four_peaks_simple_results, "Four Peaks (simple)", "time")

# Plot Fitness vs Function Evaluations (FEvals)
# plot_results(combined_four_peaks_simple_results, "Four Peaks (simple)", "fevals")
