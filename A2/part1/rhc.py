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
from problems import (
    tsp_simple_problem,
    tsp_complex_problem,
    knapsack_simple_problem,
    knapsack_complex_problem,
)

# ===== 2. Define and Run the Runner =====


# Define runner for Randomized Hill Climbing (RHC)
def run_rhc(problem, p_name):
    runner = mlrose.runners.RHCRunner(
        problem=problem,
        experiment_name=f"rhc-{p_name}",
        output_directory="./optimization_results",
        seed=42,
        iteration_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        restart_list=[5],  # This will give 5 rounds of data (5 series in the plot)
        max_attempts=MAX_ATTEMPTS,
    )
    return runner.run()


# # Four peaks
# rhc_four_peaks_simple_results = run_rhc(
#     four_peaks_simple_problem, p_name="four_peaks-simple"
# )

# (Simple Version)
# Traveling Salesman Problem
rhc_tsp_simple_results = run_rhc(
    tsp_simple_problem, p_name="TSP-simple"
)

# Knapsack
rhc_knapsack_simple_results = run_rhc(
  knapsack_simple_problem, p_name="knapsack-simple"
)

# (Complicated Version)
rhc_tsp_complex_results = run_rhc(tsp_complex_problem, p_name="TSP-complex")

# Knapsack
rhc_knapsack_complex_results = run_rhc(
    knapsack_complex_problem, p_name="knapsack-complex"
)


# ===== 3. Plot the results =====


# TODO: finish the comparision
def plot_rhc_results(results, problem_name):
    file_path = f"./optimization_results/rhc-{problem_name}/rhc__rhc-{problem_name}__run_stats_df.csv"
    rhc_results = pd.read_csv(file_path)

    # Filter data for plotting
    restart_value = rhc_results["current_restart"].unique()

    # Plot Iteration vs Fitness
    plt.figure(figsize=(14, 12))
    plt.subplot(3, 1, 1)
    for restart in restart_value:
        restart_data = rhc_results[rhc_results["current_restart"] == restart][:10]
        plt.plot(restart_data["Iteration"], restart_data["Fitness"], label=f"restart {restart}")
    plt.title(f"Iteration vs Fitness for RHC ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()

    # Plot Iteration vs Time
    plt.subplot(3, 1, 2)
    for restart in restart_value:
        restart_data = rhc_results[rhc_results["current_restart"] == restart][:10]
        plt.plot(restart_data["Iteration"], restart_data["Time"], label=f"restart {restart}")
    plt.title(f"Iteration vs Time for RHC ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.legend()

    # Plot Iteration vs Function Evaluations (FEvals)
    plt.subplot(3, 1, 3)
    for restart in restart_value:
        restart_data = rhc_results[rhc_results["current_restart"] == restart][:10]
        plt.plot(restart_data["Iteration"], restart_data["FEvals"], label=f"restart {restart}")
    plt.title(f"Iteration vs Function Evaluations for RHC ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Function Evaluations")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Simple
plot_rhc_results(rhc_tsp_simple_results, "TSP-simple")
plot_rhc_results(rhc_knapsack_simple_results, "knapsack-simple")


# Complex
plot_rhc_results(rhc_tsp_complex_results, "TSP-complex")
plot_rhc_results(rhc_knapsack_complex_results, "knapsack-complex")
