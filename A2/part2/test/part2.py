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
        experiment_name="rhc_nn",
        output_directory="./rhc_nn_results",
        seed=42,
        iteration_list=[10, 100, 500, 1000],
        max_attempts=100,
        generate_curves=True,
        algorithm="random_hill_climb",
        grid_search_parameters={"restarts": [0]},
    )

    # Run the optimization
    results = runner.run()

    return results


# Run the RHC experiment
rhc_results = run_rhc_nn(X_train, y_train)


# Plotting function for one algorithm
def plot_nn_results(results, title):
    results_df = results[0].RunHistory

    plt.figure(figsize=(14, 6))

    # Plot Iteration vs Fitness
    plt.subplot(1, 3, 1)
    for label, df in results_df.groupby("Restart"):
        plt.plot(df["Iteration"], df["Fitness"], label=f"Restarts: {label}")
    plt.title(f"Iteration vs Fitness for {title}")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.legend()

    # Plot Iteration vs Time
    plt.subplot(1, 3, 2)
    for label, df in results_df.groupby("Restart"):
        plt.plot(df["Iteration"], df["Time"], label=f"Restarts: {label}")
    plt.title(f"Iteration vs Time for {title}")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.legend()

    # Plot Iteration vs Function Evaluations (FEvals)
    plt.subplot(1, 3, 3)
    for label, df in results_df.groupby("Restart"):
        plt.plot(df["Iteration"], df["FEvals"], label=f"Restarts: {label}")
    plt.title(f"Iteration vs Function Evaluations for {title}")
    plt.xlabel("Iteration")
    plt.ylabel("Function Evaluations")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Plot results for RHC
plot_nn_results(rhc_results, "RHC")
