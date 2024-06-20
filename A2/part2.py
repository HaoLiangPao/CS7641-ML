import mlrose_hiive as mlrose
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and train the neural network with RHC
nn_rhc = mlrose.NeuralNetwork(
    hidden_nodes=[64, 64],
    activation="relu",
    algorithm="random_hill_climb",
    max_iters=1000,
    bias=True,
    is_classifier=True,
    learning_rate=0.001,
    early_stopping=True,
    clip_max=1e10,
    restarts=15,  # Exclusive to RHC
    # schedule=mlrose.GeomDecay(),  # Exclusive to SA
    # pop_size=200,  # Exclusive to GA
    max_attempts=100,
    random_state=42,
    curves=True,
)
nn_rhc.fit(X_train, y_train)

# Predict and evaluate the model
y_train_pred = nn_rhc.predict(X_train)
y_test_pred = nn_rhc.predict(X_test)

# Convert one-hot encoded predictions back to class labels
y_train_pred = np.argmax(y_train_pred, axis=1)
y_test_pred = np.argmax(y_test_pred, axis=1)
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"RHC Train Accuracy: {train_accuracy}")
print(f"RHC Test Accuracy: {test_accuracy}")
