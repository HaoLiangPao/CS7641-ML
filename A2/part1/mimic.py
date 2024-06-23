import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json5

# Customized Libraries
from utils import CustomKnapsack, plot_op_results

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


# Define runner for MIMIC
def run_mimic(problem, p_name, iteration_list, population_sizes, keep_percent_list):
    runner = mlrose.runners.MIMICRunner(
        problem=problem,
        experiment_name=f"mimic-{p_name}",
        output_directory="./optimization_results",
        seed=42,
        iteration_list=iteration_list,
        max_attempts=MAX_ATTEMPTS,
        population_sizes=population_sizes,
        keep_percent_list=keep_percent_list,
        use_fast_mimic=True,
    )
    return runner.run()


# # Four peaks
# ga_four_peaks_simple_results = run_ga(
#     four_peaks_simple_problem, p_name="four_peaks-simple"
# )

iteration_list = range(0, 100, 5)
# # # 1. fixed population
# population_sizes = [10]
# keep_percent_list = [0.1, 0.2, 0.3, 0.4]

# 2. fixed percentage
population_sizes = range(10, 200, 50)
keep_percent_list = [0.3]

# Traveling Salesman Problem
mimic_tsp_simple_results = run_mimic(
    tsp_simple_problem,
    p_name="TSP-simple",
    iteration_list=iteration_list,
    population_sizes=population_sizes,
    keep_percent_list=keep_percent_list,
)

# Knapsack
mimic_knapsack_simple_results = run_mimic(
    knapsack_simple_problem,
    p_name="knapsack-simple",
    iteration_list=iteration_list,
    population_sizes=population_sizes,
    keep_percent_list=keep_percent_list,
)

# Traveling Salesman Problem
mimic_tsp_complex_results = run_mimic(
    tsp_complex_problem,
    p_name="TSP-complex",
    iteration_list=iteration_list,
    population_sizes=population_sizes,
    keep_percent_list=keep_percent_list,
)

# Knapsack
mimic_knapsack_complex_results = run_mimic(
    knapsack_complex_problem,
    p_name="knapsack-complex",
    iteration_list=iteration_list,
    population_sizes=population_sizes,
    keep_percent_list=keep_percent_list,
)
# ===== 3. Plot the results =====


# TODO: finish the comparision
def plot_mimic_results(results, problem_name, attribute_col, attribute):
    file_path = f"./optimization_results/mimic-{problem_name}/mimic__mimic-{problem_name}__run_stats_df.csv"
    mimic_results = pd.read_csv(file_path)

    # Filter data for plotting
    attribute_col_value = mimic_results[attribute_col].unique()

    # Plot Iteration vs Fitness
    plt.figure(figsize=(14, 6))
    for attribute_value in attribute_col_value:
        attribute_value_data = mimic_results[
            mimic_results[attribute_col] == attribute_value
        ][: len(iteration_list)]
        plt.plot(
            attribute_value_data["Iteration"],
            attribute_value_data["Fitness"],
            label=f"{attribute} {attribute_value}",
        )
    plt.title(f"Iteration vs Fitness for MIMIC ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

    # Plot Iteration vs Time
    plt.figure(figsize=(14, 6))
    for attribute_value in attribute_col_value:
        attribute_value_data = mimic_results[
            mimic_results[attribute_col] == attribute_value
        ][: len(iteration_list)]
        plt.plot(
            attribute_value_data["Iteration"],
            attribute_value_data["Time"],
            label=f"{attribute} {attribute_value}",
        )
    plt.title(f"Iteration vs Time for MIMIC ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

    # Plot Iteration vs Function Evaluations (FEvals)
    plt.figure(figsize=(14, 6))
    for attribute_value in attribute_col_value:
        attribute_value_data = mimic_results[
            mimic_results[attribute_col] == attribute_value
        ][: len(iteration_list)]
        plt.plot(
            attribute_value_data["Iteration"],
            attribute_value_data["FEvals"],
            label=f"{attribute} {attribute_value}",
        )
    plt.title(f"Iteration vs Function Evaluations for MIMIC ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Function Evaluations")
    plt.legend()
    plt.show()

# 1. fixed population
# plot_op_results(
#     results=mimic_tsp_simple_results,
#     problem_name="TSP-simple",
#     attribute_col="Keep Percent",
#     attribute="keep_percentage",
#     algorithm="mimic",
#     iteration_list=iteration_list,
# )
# plot_op_results(
#     results=mimic_knapsack_simple_results,
#     problem_name="knapsack-simple",
#     attribute_col="Keep Percent",
#     attribute="keep_percentage",
#     algorithm="mimic",
#     iteration_list=iteration_list,
# )

# plot_op_results(
#     results=mimic_tsp_complex_results,
#     problem_name="TSP-complex",
#     attribute_col="Keep Percent",
#     attribute="keep_percentage",
#     algorithm="mimic",
#     iteration_list=iteration_list,
# )
# plot_op_results(
#     results=mimic_knapsack_complex_results,
#     problem_name="knapsack-complex",
#     attribute_col="Keep Percent",
#     attribute="keep_percentage",
#     algorithm="mimic",
#     iteration_list=iteration_list,
# )

# 2. fixed percentage
plot_op_results(
    results=mimic_tsp_simple_results,
    problem_name="TSP-simple",
    attribute_col="Population Size",
    attribute="population",
    algorithm="mimic",
    iteration_list=iteration_list,
)
plot_op_results(
    results=mimic_knapsack_simple_results,
    problem_name="knapsack-simple",
    attribute_col="Population Size",
    attribute="population",
    algorithm="mimic",
    iteration_list=iteration_list,
)

plot_op_results(
    results=mimic_tsp_complex_results,
    problem_name="TSP-complex",
    attribute_col="Population Size",
    attribute="population",
    algorithm="mimic",
    iteration_list=iteration_list,
)
plot_op_results(
    results=mimic_knapsack_complex_results,
    problem_name="knapsack-complex",
    attribute_col="Population Size",
    attribute="population",
    algorithm="mimic",
    iteration_list=iteration_list,
)
