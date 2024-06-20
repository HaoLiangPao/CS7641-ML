import mlrose_hiive as mlrose
import numpy as np
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

# Simple Version
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

# Complicated Version
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

# ===== 2. Implement Randomized Optimization algorithms =====


# Randomized Hill Climbing (RHC)
def run_rhc(problem, max_attempts, max_iters, restarts, num_runs):
    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(
        problem,
        max_attempts=max_attempts,
        max_iters=max_iters,
        restarts=restarts,
        curve=True,
    )
    return best_state, best_fitness, fitness_curve


# Simulated Annealing (SA)
def run_sa(problem, schedule, max_attempts, max_iters):
    best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
        problem,
        schedule=eval(f"mlrose.{schedule}"),
        max_attempts=max_attempts,
        max_iters=max_iters,
        curve=True,
    )
    return best_state, best_fitness, fitness_curve


# Genetic Algorithm (GA)
def run_ga(problem, pop_size, mutation_prob, max_attempts, max_iters):
    best_state, best_fitness, fitness_curve = mlrose.genetic_alg(
        problem,
        pop_size=pop_size,
        mutation_prob=mutation_prob,
        max_attempts=max_attempts,
        max_iters=max_iters,
        curve=True,
    )
    return best_state, best_fitness, fitness_curve


# Mutual-Information-Maximizing Input Clustering (MIMIC)
def run_mimic(problem, pop_size, keep_pct, max_attempts, max_iters):
    best_state, best_fitness, fitness_curve = mlrose.mimic(
        problem,
        pop_size=pop_size,
        keep_pct=keep_pct,
        max_attempts=max_attempts,
        max_iters=max_iters,
        curve=True,
    )
    return best_state, best_fitness, fitness_curve


# ===== 3. Run the algorithms and collect performance data =====
def plot_performance(problem, problem_name):
    plt.figure()

    # Run Randomized Hill Climbing
    rhc_state, rhc_fitness, rhc_curve = run_rhc(
        problem=problem,
        max_attempts=MAX_ATTEMPTS,
        max_iters=MAX_ITERS,
        restarts=config["rhc"]["restarts"],
        num_runs=config["rhc"]["num_runs"],
    )
    plt.plot([value[0] for value in rhc_curve], label="RHC")

    # Run Simulated Annealing
    # scheule = mlrose.GeomDecay()
    # if config["sa"]["schedule"] == "ExpDecay()":
    #     schedule = mlrose.ExpDecay()
    sa_state, sa_fitness, sa_curve = run_sa(
        problem=problem,
        max_attempts=MAX_ATTEMPTS,
        max_iters=MAX_ITERS,
        schedule=config["sa"]["schedule"],
    )
    plt.plot([value[0] for value in sa_curve], label="SA")

    # Run Genetic Algorithm
    ga_state, ga_fitness, ga_curve = run_ga(
        problem=problem,
        max_attempts=MAX_ATTEMPTS,
        max_iters=MAX_ITERS,
        pop_size=config["ga"]["pop_size"],
        mutation_prob=config["ga"]["mutation_prob"],
    )
    plt.plot([value[0] for value in ga_curve], label="GA")

    # Run MIMIC
    mimic_state, mimic_fitness, mimic_curve = run_mimic(
        problem=problem,
        max_attempts=MAX_ATTEMPTS,
        max_iters=MAX_ITERS,
        pop_size=config["mimic"]["pop_size"],
        keep_pct=config["mimic"]["keep_pct"],
    )
    plt.plot([value[0] for value in mimic_curve], label="MIMIC")

    # Plot the performance curves
    plt.title(f"Performance Comparison for {problem_name}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()


plot_performance(four_peaks_simple_problem, "Four Peaks (simple)")
plot_performance(knapsack_simple_problem, "Knapsack (simple)")

plot_performance(four_peaks_problem, "Four Peaks (complicate)")
plot_performance(knapsack_problem, "Knapsack (complicate)")
