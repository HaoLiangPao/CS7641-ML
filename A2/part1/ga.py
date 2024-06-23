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


# Define runner for Genetic Algorithm (GA)
def run_ga(problem, p_name, iteration_list, population_sizes, mutation_rates):
    runner = mlrose.runners.GARunner(
        problem=problem,
        experiment_name=f"ga-{p_name}",
        output_directory="./optimization_results",
        seed=42,
        iteration_list=iteration_list,
        max_attempts=MAX_ATTEMPTS,
        population_sizes=population_sizes,
        mutation_rates=mutation_rates,
    )
    return runner.run()

iteration_list = range(0, 1000, 50)

# # 1. fixed population
# population_sizes = [10]
# mutation_rates = [0.1, 0.2, 0.3, 0.4]

# 2. fixed mutation_rates
population_sizes = range(10, 200, 50)
mutation_rates = [0.1]

# Traveling Salesman Problem
ga_tsp_simple_results = run_ga(
    tsp_simple_problem,
    p_name="TSP-simple",
    iteration_list=iteration_list,
    population_sizes=population_sizes,
    mutation_rates=mutation_rates,
)

ga_tsp_complex_results = run_ga(
    tsp_complex_problem,
    p_name="TSP-complex",
    iteration_list=iteration_list,
    population_sizes=population_sizes,
    mutation_rates=mutation_rates,
)

# Knapsack
ga_knapsack_simple_results = run_ga(
    knapsack_simple_problem,
    p_name="knapsack-simple",
    iteration_list=iteration_list,
    population_sizes=population_sizes,
    mutation_rates=mutation_rates,
)

ga_knapsack_complex_results = run_ga(
    knapsack_complex_problem,
    p_name="knapsack-complex",
    iteration_list=iteration_list,
    population_sizes=population_sizes,
    mutation_rates=mutation_rates,
)
# ===== 3. Plot the results =====

# TODO: finish the comparision
def plot_ga_results(results, problem_name):
    file_path = f"./optimization_results/ga-{problem_name}/ga__ga-{problem_name}__run_stats_df.csv"
    ga_results = pd.read_csv(file_path)

    # Filter data for plotting
    mutation_value = ga_results["Mutation Rate"].unique()

    # Plot Iteration vs Fitness
    plt.figure(figsize=(14, 6))
    for mutation_rate in mutation_value:
        mutation_rate_data = ga_results[ga_results["Mutation Rate"] == mutation_rate][:len(iteration_list)]
        plt.plot(mutation_rate_data["Iteration"], mutation_rate_data["Fitness"], label=f"mutation_rate {mutation_rate}")
    plt.title(f"Iteration vs Fitness for GA ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()

    # Plot Iteration vs Time
    plt.figure(figsize=(14, 6))
    for mutation_rate in mutation_value:
        mutation_rate_data = ga_results[ga_results["Mutation Rate"] == mutation_rate][:len(iteration_list)]
        plt.plot(mutation_rate_data["Iteration"], mutation_rate_data["Time"], label=f"mutation_rate {mutation_rate}")
    plt.title(f"Iteration vs Time for GA ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

    # Plot Iteration vs Function Evaluations (FEvals)
    plt.figure(figsize=(14, 6))
    for mutation_rate in mutation_value:
        mutation_rate_data = ga_results[ga_results["Mutation Rate"] == mutation_rate][:len(iteration_list)]
        plt.plot(mutation_rate_data["Iteration"], mutation_rate_data["FEvals"], label=f"mutation_rate {mutation_rate}")
    plt.title(f"Iteration vs Function Evaluations for GA ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Function Evaluations")
    plt.legend()
    plt.show()

# # 1. fixed population
# plot_op_results(
#     results=ga_tsp_simple_results,
#     problem_name="TSP-simple",
#     attribute_col="Mutation Rate",
#     attribute="mutation_rate",
#     algorithm="ga",
#     iteration_list=iteration_list,
# )
# plot_op_results(
#     results=ga_knapsack_simple_results,
#     problem_name="knapsack-simple",
#     attribute_col="Mutation Rate",
#     attribute="mutation_rate",
#     algorithm="ga",
#     iteration_list=iteration_list,
# )

# plot_op_results(
#     results=ga_tsp_complex_results,
#     problem_name="TSP-complex",
#     attribute_col="Mutation Rate",
#     attribute="mutation_rate",
#     algorithm="ga",
#     iteration_list=iteration_list,
# )
# plot_op_results(
#     results=ga_knapsack_complex_results,
#     problem_name="knapsack-complex",
#     attribute_col="Mutation Rate",
#     attribute="mutation_rate",
#     algorithm="ga",
#     iteration_list=iteration_list,
# )

# 2. fixed mutation_rates
plot_op_results(
    results=ga_tsp_simple_results,
    problem_name="TSP-simple",
    attribute_col="Population Size",
    attribute="population",
    algorithm="ga",
    iteration_list=iteration_list
)

plot_op_results(
    results=ga_knapsack_simple_results,
    problem_name="knapsack-simple",
    attribute_col="Population Size",
    attribute="population",
    algorithm="ga",
    iteration_list=iteration_list,
)

plot_op_results(
    results=ga_tsp_complex_results,
    problem_name="TSP-complex",
    attribute_col="Population Size",
    attribute="population",
    algorithm="ga",
    iteration_list=iteration_list,
)
plot_op_results(
    results=ga_knapsack_complex_results,
    problem_name="knapsack-complex",
    attribute_col="Population Size",
    attribute="population",
    algorithm="ga",
    iteration_list=iteration_list,
)
