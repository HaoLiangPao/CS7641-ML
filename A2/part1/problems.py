import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import random


# == Simple Version ==
# Four Peaks Problem (Maximization, favors SA)
four_peaks_fitness = mlrose.FourPeaks(t_pct=0.15)
four_peaks_simple_problem = mlrose.DiscreteOpt(
    length=20, fitness_fn=four_peaks_fitness, maximize=True, max_val=2
)

# Traveling Salesman Problem (Minimization, favors SA)
# Define a list of coordinates for each city
coords_list = [(0, 0), (1, 5), (2, 3), (5, 2), (6, 6)]

# Create the fitness function for TSP
tsp_fitness = mlrose.TravellingSales(coords=coords_list)
tsp_simple_problem = mlrose.TSPOpt(
    length=len(coords_list), fitness_fn=tsp_fitness, maximize=False
)

# Knapsack Problem (Maximization, favors GA)
weights = [10, 5, 2, 8, 15, 7, 3, 20, 6, 1]
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

knapsack_fitness = mlrose.Knapsack(weights, values, max_weight_pct=0.6)
knapsack_simple_problem = mlrose.DiscreteOpt(
    length=10, fitness_fn=knapsack_fitness, maximize=True, max_val=2
)

# == Complicated Version ==

# Generate a large list of random coordinates
random.seed(42)
coords_list = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(50)]

# Traveling Salesman Problem (Minimization, favors SA)
tsp_complex_fitness = mlrose.TravellingSales(coords=coords_list)
tsp_complex_problem = mlrose.TSPOpt(
    length=len(coords_list), fitness_fn=tsp_complex_fitness, maximize=False
)

# Knapsack Problem (Maximization, favors GA)
weights = [random.randint(1, 100) for _ in range(50)]
values = [random.randint(1, 100) for _ in range(50)]

knapsack_complex_fitness = mlrose.Knapsack(weights, values, max_weight_pct=0.5)
knapsack_complex_problem = mlrose.DiscreteOpt(
    length=len(weights), fitness_fn=knapsack_complex_fitness, maximize=True, max_val=2
)
