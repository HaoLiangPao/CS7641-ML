import mlrose_hiive as mlrose
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load and process the wine quality dataset
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red_wine = pd.read_csv(red_wine_url, delimiter=";")
white_wine = pd.read_csv(white_wine_url, delimiter=";")

red_wine["type"] = 0
white_wine["type"] = 1

wine_data = pd.concat([red_wine, white_wine])

# Process the wine dataset
X = wine_data.drop(["quality"], axis=1)
y = wine_data["quality"].values

# One-hot encode the labels
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define and run the neural network with RHC using the Runner class
def run_rhc_nn(X_train, y_train):
    nn_model = mlrose.NeuralNetwork(
        hidden_nodes=[64, 64],
        activation="relu",
        algorithm="random_hill_climb",
        max_iters=1000,
        bias=True,
        is_classifier=True,
        learning_rate=0.001,
        early_stopping=True,
        clip_max=1e10,
        max_attempts=100,
        random_state=42,
        curve=True,
    )

    runner = mlrose.runners.NNGARunner(
        problem=nn_model,
        experiment_name="rhc_nn",
        output_directory="./rhc_nn_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=100,
    )
    results = runner.run()
    return results


# Define and run the neural network with SA using the Runner class
def run_sa_nn(X_train, y_train):
    nn_model = mlrose.NeuralNetwork(
        hidden_nodes=[64, 64],
        activation="relu",
        algorithm="simulated_annealing",
        max_iters=1000,
        bias=True,
        is_classifier=True,
        learning_rate=0.001,
        early_stopping=True,
        clip_max=1e10,
        max_attempts=100,
        random_state=42,
        curve=True,
    )

    runner = mlrose.runners.NNGARunner(
        problem=nn_model,
        experiment_name="sa_nn",
        output_directory="./sa_nn_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=100,
    )
    results = runner.run()
    return results


# Define and run the neural network with GA using the Runner class
def run_ga_nn(X_train, y_train):
    nn_model = mlrose.NeuralNetwork(
        hidden_nodes=[64, 64],
        activation="relu",
        algorithm="genetic_alg",
        max_iters=1000,
        bias=True,
        is_classifier=True,
        learning_rate=0.001,
        early_stopping=True,
        clip_max=1e10,
        max_attempts=100,
        random_state=42,
        curve=True,
    )

    runner = mlrose.runners.NNGARunner(
        problem=nn_model,
        experiment_name="ga_nn",
        output_directory="./ga_nn_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=100,
        population_sizes=[200],
        mutation_rates=[0.1],
    )
    results = runner.run()
    return results


# Define and run the neural network with MIMIC using the Runner class
def run_mimic_nn(X_train, y_train):
    nn_model = mlrose.NeuralNetwork(
        hidden_nodes=[64, 64],
        activation="relu",
        algorithm="mimic",
        max_iters=1000,
        bias=True,
        is_classifier=True,
        learning_rate=0.001,
        early_stopping=True,
        clip_max=1e10,
        max_attempts=100,
        random_state=42,
        curve=True,
    )

    runner = mlrose.runners.NNGARunner(
        problem=nn_model,
        experiment_name="mimic_nn",
        output_directory="./mimic_nn_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=100,
        population_sizes=[200],
        keep_percent_list=[0.1, 0.2, 0.3],
        use_fast_mimic=True,
    )
    results = runner.run()
    return results


# Run experiments
rhc_results = run_rhc_nn(X_train, y_train)
sa_results = run_sa_nn(X_train, y_train)
ga_results = run_ga_nn(X_train, y_train)
mimic_results = run_mimic_nn(X_train, y_train)


# Plotting function for one algorithm
def plot_nn_results(results, title):
    results_df = results["Fitness_Curves"]

    plt.figure(figsize=(14, 6))

    # Plot Iteration vs Fitness
    plt.subplot(1, 3, 1)
    plt.plot(results_df["Iteration"], results_df["Fitness"])
    plt.title(f"Iteration vs Fitness for {title}")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")

    # Plot Iteration vs Time
    plt.subplot(1, 3, 2)
    plt.plot(results_df["Iteration"], results_df["Time"])
    plt.title(f"Iteration vs Time for {title}")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")

    # Plot Iteration vs Function Evaluations (FEvals)
    plt.subplot(1, 3, 3)
    plt.plot(results_df["Iteration"], results_df["FEvals"])
    plt.title(f"Iteration vs FEvals for {title}")
    plt.xlabel("Iteration")
    plt.ylabel("Function Evaluations")

    plt.tight_layout()
    plt.show()


# Plot results for each algorithm
plot_nn_results(rhc_results, "RHC")
plot_nn_results(sa_results, "SA")
plot_nn_results(ga_results, "GA")
plot_nn_results(mimic_results, "MIMIC")
