import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt

# ===== 1. Define optimization problems =====
# Four Peaks Problem
four_peaks_fitness = mlrose.FourPeaks(t_pct=0.15)
four_peaks_problem = mlrose.DiscreteOpt(
    length=20,
    fitness_fn=four_peaks_fitness,
    maximize=True,
    max_val=2
)

# Knapsack Problem
weights = [10, 5, 2, 8, 15, 7, 3, 20, 6, 1]
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
knapsack_fitness = mlrose.Knapsack(weights, values, max_weight_pct=0.6)
knapsack_problem = mlrose.DiscreteOpt(
    length=10, fitness_fn=knapsack_fitness, maximize=True, max_val=2
)

# ===== 2. Implement Randomized Optimization algorithms =====

# Randomized Hill Climbing (RHC)
def run_rhc(problem, max_attempts=10, max_iters=1000, restarts=0, num_runs=10):
    best_fitnesses = []
    best_states = []
    for run in range(num_runs):
        best_state, best_fitness, _ = mlrose.random_hill_climb(
            problem, max_attempts=max_attempts, max_iters=max_iters, restarts=restarts
        )
        best_fitnesses.append(best_fitness)
        print(f"[RHC] states are {best_states}")
    return (best_fitnesses, best_states)

# Simulated Annealing (SA)
def run_sa(problem, schedule=mlrose.ExpDecay(), max_attempts=10, max_iters=1000):
    best_state, best_fitness, _ = mlrose.simulated_annealing(
        problem, schedule=schedule, max_attempts=max_attempts, max_iters=max_iters
    )
    return best_state, best_fitness

# Genetic Algorithm (GA)
def run_ga(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=1000):
    best_state, best_fitness, _ = mlrose.genetic_alg(
        problem,
        pop_size=pop_size,
        mutation_prob=mutation_prob,
        max_attempts=max_attempts,
        max_iters=max_iters,
    )
    return best_state, best_fitness


# ===== 3. Run the algorithm =====
# a) four peaks
rhc_four_peaks_states, rhc_four_peaks_fitnesses = run_rhc(
    four_peaks_problem,
    max_attempts=100,
    max_iters=1000,
    restarts=0,
    num_runs=num_runs
)
print("[RHC] Four Peaks Best states:", rhc_four_peaks_states)
print("[RHC] Four Peaks Best fitnesses:", rhc_four_peaks_fitnesses)

schedule = mlrose.ExpDecay()  # Exponential decay schedule
best_state, best_fitness = run_sa(
    four_peaks_problem,
    schedule=schedule,
    max_attempts=100,
    max_iters=1000
)
print("[SA] Four Peaks Best state:", best_state)
print("[SA] Four Peaks Best fitness:", best_fitness)

best_state, best_fitness = run_ga(
    four_peaks_problem,
    pop_size=200,
    mutation_prob=0.1,
    max_attempts=100,
    max_iters=1000,
)
print("[GA] Four Peaks Best state:", best_state)
print("[GA] Four Peaks Best fitness:", best_fitness)

# # b) knapsack
# rhc_knapsack_results = run_rhc(
#     knapsack_problem,
#     max_attempts=100,
#     max_iters=1000,
#     restarts=0,
#     num_runs=num_runs
# )
# print("Knapsack Results:", rhc_knapsack_results)


# # ===== 4. Analysis =====

# def plot_results(results, title):
#     plt.figure(figsize=(10, 6))
#     plt.hist(results, bins=10, alpha=0.7)
#     plt.title(f"{title} Fitness Distribution")
#     plt.xlabel("Fitness")
#     plt.ylabel("Frequency")
#     plt.grid(True)
#     plt.show()


# # Plot results for Four Peaks
# plot_results(rhc_four_peaks_results, "Four Peaks")

# # Plot results for Knapsack
# plot_results(rhc_knapsack_results, "Knapsack")


# import time


# def evaluate_rhc(problem, max_attempts=10, max_iters=1000, restarts=0):
#     start_time = time.time()
#     best_state, best_fitness, _ = mlrose.random_hill_climb(
#         problem, max_attempts=max_attempts, max_iters=max_iters, restarts=restarts
#     )
#     end_time = time.time()
#     duration = end_time - start_time
#     return best_state, best_fitness, duration


# # Evaluate on Four Peaks
# four_peaks_state, four_peaks_fitness, four_peaks_time = evaluate_rhc(
#     four_peaks_problem, max_attempts=100, max_iters=1000, restarts=0
# )
# print("Four Peaks - Best State:", four_peaks_state)
# print("Four Peaks - Best Fitness:", four_peaks_fitness)
# print("Four Peaks - Duration (seconds):", four_peaks_time)

# # Evaluate on Knapsack
# knapsack_state, knapsack_fitness, knapsack_time = evaluate_rhc(
#     knapsack_problem, max_attempts=100, max_iters=1000, restarts=0
# )
# print("Knapsack - Best State:", knapsack_state)
# print("Knapsack - Best Fitness:", knapsack_fitness)
# print("Knapsack - Duration (seconds):", knapsack_time)
