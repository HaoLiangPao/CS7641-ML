import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json5

# Customized Libraries
from utils import CustomKnapsack

# Load configuration
with open("configs.json5") as f:
    config = json5.load(f)

MAX_ATTEMPTS = config["shared"]["max_attempts"]
MAX_ITERS = config["shared"]["max_iters"]

# ===== 1. Define optimization problems =====

# == Simple Version ==
# Four Peaks Problem (Maximization, favors SA)
four_peaks_fitness = mlrose.FourPeaks(t_pct=0.15)
four_peaks_simple_problem = mlrose.DiscreteOpt(
    length=20,
    fitness_fn=four_peaks_fitness, 
    maximize=True,
    max_val=2
)


# Custom Knapsack Problem (Minimization, favors GA)
weights = [10, 5, 2, 8, 15, 7, 3, 20, 6, 1]
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
knapsack_fitness = CustomKnapsack(weights, values, max_weight_pct=0.6, target_weight_pct=0.5)
knapsack_simple_problem = mlrose.DiscreteOpt(
    length=10,
    fitness_fn=knapsack_fitness,
    maximize=False,
    max_val=2
)

# == Complicated Version ==
# Four Peaks Problem
four_peaks_fitness = mlrose.FourPeaks(t_pct=0.15)
four_peaks_problem = mlrose.DiscreteOpt(
    length=40,  # Increased length for more complexity
    fitness_fn=four_peaks_fitness,
    maximize=True,
    max_val=2
)

# Knapsack Problem
weights = [10, 5, 2, 8, 15, 7, 3, 20, 6, 1, 25, 12, 18, 14, 9, 11, 23, 17, 19, 21]  # Increased weights and items for more complexity
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
knapsack_fitness = CustomKnapsack(weights, values, max_weight_pct=0.6, target_weight_pct=0.5)
knapsack_problem = mlrose.DiscreteOpt(
    length=20,  # Increased length for more complexity
    fitness_fn=knapsack_fitness,
    maximize=False,
    max_val=2
)

# ===== 2. Define and Run the Runner =====


# Define runner for Randomized Hill Climbing (RHC)
def run_rhc(problem):
    runner = mlrose.runners.RHCRunner(
        problem=problem,
        experiment_name="rhc",
        output_directory="./optimization_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        restart_list=[10], # This will give 10 rounds of data (10 series in the plot)
        max_attempts=MAX_ATTEMPTS,
    )
    return runner.run()


# Define runner for Simulated Annealing (SA)
def run_sa(problem):
    runner = mlrose.runners.SARunner(
        problem=problem,
        experiment_name="sa",
        output_directory="./optimization_results",
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
        output_directory="./optimization_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=MAX_ATTEMPTS,
        population_sizes=[10],
        mutation_rates=[0.1, 0.2, 0.3, 0.4],
    )
    return runner.run()


# Define runner for MIMIC
def run_mimic(problem):
    runner = mlrose.runners.MIMICRunner(
        problem=problem,
        experiment_name="mimic",
        output_directory="./optimization_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=MAX_ATTEMPTS,
        population_sizes=[10, 50, 100, 200],
        keep_percent_list=[0.4],
        use_fast_mimic=True,
    )
    return runner.run()


# Four peaks
rhc_four_peaks_simple_results = run_rhc(four_peaks_simple_problem)
sa_four_peaks_simple_results = run_sa(four_peaks_simple_problem)
ga_four_peaks_simple_results = run_ga(four_peaks_simple_problem)
mimic_four_peaks_simple_results = run_mimic(four_peaks_simple_problem)

# Knapsack
rhc_knapsack_simple_results = run_rhc(knapsack_simple_problem)
sa_knapsack_simple_results = run_sa(knapsack_simple_problem)
ga_knapsack_simple_results = run_ga(knapsack_simple_problem)
mimic_knapsack_simple_results = run_mimic(knapsack_simple_problem)

# ===== 3. Plot the results =====


# TODO: finish the comparision
def plot_rhc_results(results, problem_name):
    pass

def plot_sa_results(results, problem_name):
    file_path = "./optimization_results/sa/sa__sa__run_stats_df.csv"
    sa_results = pd.read_csv(file_path)

    # Filter data for plotting
    temp_values = sa_results["schedule_init_temp"].unique()

    # Plot Iteration vs Fitness
    plt.figure(figsize=(14, 6))
    for temp in temp_values:
        temp_data = sa_results[sa_results["schedule_init_temp"] == temp][:5]
        plt.plot(temp_data["Iteration"], temp_data["Fitness"], label=f"Temp {temp}")
    plt.title(f"Iteration vs Fitness for SA ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

    # Plot Iteration vs Time
    plt.figure(figsize=(14, 6))
    for temp in temp_values:
        temp_data = sa_results[sa_results["schedule_init_temp"] == temp][:5]
        plt.plot(temp_data["Iteration"], temp_data["Time"], label=f"Temp {temp}")
    plt.title(f"Iteration vs Time for SA ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

    # Plot Iteration vs Function Evaluations (FEvals)
    plt.figure(figsize=(14, 6))
    for temp in temp_values:
        temp_data = sa_results[sa_results["schedule_init_temp"] == temp][:5]
        plt.plot(temp_data["Iteration"], temp_data["FEvals"], label=f"Temp {temp}")
    plt.title(f"Iteration vs Function Evaluations for SA ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Function Evaluations")
    plt.legend()
    plt.show()

# TODO: finish the comparision
def plot_ga_results(results, problem_name):
    pass

# TODO: finish the comparision
def plot_mimic_results(results, problem_name):
    pass


plot_rhc_results(sa_four_peaks_simple_results, "Simple 4 Peak")
plot_sa_results(sa_four_peaks_simple_results, "Simple 4 Peak")
plot_ga_results(sa_four_peaks_simple_results, "Simple 4 Peak")
plot_mimic_results(sa_four_peaks_simple_results, "Simple 4 Peak")


# def plot_results(results, problem_name, metric):
#     plt.figure(figsize=(14, 6))

#     for label, result in results.items():
#         print(result)
#         print(len(result))
#         print(type(result))

#         print(result.iloc[:, 2].values)
#         print(len(result.iloc[:, 2].values))
#         print(type(result.iloc[:, 2].values))

#         print(label)

#         iteration = result.iloc[:, 0].values
#         fitness = result.iloc[:, 1].values
#         fevals = result.iloc[:, 2].values
#         time = result.iloc[:, 3].values
#         population_size = result.iloc[:, 4].values
#         mutation_rate = result.iloc[:, 5].values
#         max_iters = result.iloc[:, 6].values

#         if metric == "time":
#             plt.plot(time, fitness, label=label)
#         elif metric == "fevals":
#             plt.plot(fevals, fitness, label=label)

#     plt.title(f"{metric.capitalize()} vs Fitness for {problem_name}")
#     plt.xlabel(metric.capitalize())
#     plt.ylabel("Fitness")
#     plt.legend()
#     plt.show()

# # Combine results
# combined_four_peaks_simple_results = {
#     "RHC": rhc_four_peaks_simple_results[0],
#     # "SA": sa_four_peaks_simple_results[0],
#     # "GA": ga_four_peaks_simple_results[0],
#     # "MIMIC": mimic_four_peaks_simple_results[0],
# }

# # Plot Fitness vs Time
# plot_results(combined_four_peaks_simple_results, "Four Peaks (simple)", "time")

# # Plot Fitness vs Function Evaluations (FEvals)
# # plot_results(combined_four_peaks_simple_results, "Four Peaks (simple)", "fevals")


# Get the best hyper parameter then compare the performance on the problem.
