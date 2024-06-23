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


# Define runner for Simulated Annealing (SA)
def run_sa(problem, p_name, iteration_list, temperature_list):
    runner = mlrose.runners.SARunner(
        problem=problem,
        experiment_name=f"sa-{p_name}",
        output_directory="./optimization_results",
        seed=42,
        iteration_list=iteration_list,
        max_attempts=MAX_ATTEMPTS,
        temperature_list=temperature_list,
    )
    return runner.run()

iteration_list = range(0, 1300, 50)
# # 1. fixed population
temperature_list = [1, 10, 50, 100]

# Traveling Salesman Problem
sa_tsp_simple_results = run_sa(
    problem=tsp_simple_problem,
    p_name="TSP-simple",
    iteration_list=iteration_list,
    temperature_list=temperature_list,
)

sa_tsp_complex_results = run_sa(
    problem=tsp_complex_problem,
    p_name="TSP-complex",
    iteration_list=iteration_list,
    temperature_list=temperature_list,
)

# Knapsack
sa_knapsack_simple_results = run_sa(
    problem=knapsack_simple_problem,
    p_name="knapsack-simple",
    iteration_list=iteration_list,
    temperature_list=temperature_list,
)

sa_knapsack_complex_results = run_sa(
    problem=knapsack_complex_problem,
    p_name="knapsack-complex",
    iteration_list=iteration_list,
    temperature_list=temperature_list,
)

# ===== 3. Plot the results =====


# TODO: finish the comparision
def plot_sa_results(results, problem_name):
    file_path = f"./optimization_results/sa-{problem_name}/sa__sa-{problem_name}__run_stats_df.csv"
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


# plot_sa_results(sa_tsp_simple_results, "TSP-simple")
# plot_sa_results(sa_knapsack_simple_results, "knapsack-simple")

plot_op_results(
    results=sa_tsp_simple_results,
    problem_name="TSP-simple",
    attribute_col="schedule_init_temp",
    attribute="temp",
    algorithm="sa",
    iteration_list=iteration_list,
)

plot_op_results(
    results=sa_knapsack_simple_results,
    problem_name="knapsack-simple",
    attribute_col="schedule_init_temp",
    attribute="temp",
    algorithm="sa",
    iteration_list=iteration_list,
)


plot_op_results(
    results=sa_tsp_complex_results,
    problem_name="TSP-complex",
    attribute_col="schedule_init_temp",
    attribute="temp",
    algorithm="sa",
    iteration_list=iteration_list,
)

plot_op_results(
    results=sa_knapsack_complex_results,
    problem_name="knapsack-complex",
    attribute_col="schedule_init_temp",
    attribute="temp",
    algorithm="sa",
    iteration_list=iteration_list,
)
