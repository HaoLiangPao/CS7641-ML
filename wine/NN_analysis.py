import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

# Customized Libraries


# Load configuration from JSON file
with open("../analysis_configs.json", "r") as f:
    config = json.load(f)

# Extract hyperparameters for Wine dataset NN
wine_nn_config = config["wine"]["NN"]

TEST_SIZE = wine_nn_config["TEST_SIZE"]
RANDOM_STATE = wine_nn_config["RANDOM_STATE"]
EPOCHS = wine_nn_config["EPOCHS"]
BATCH_SIZE = wine_nn_config["BATCH_SIZE"]
VALIDATION_SPLIT = wine_nn_config["VALIDATION_SPLIT"]
ACTIVATION = wine_nn_config["ACTIVATION"]
BINARY_ACTIVATION = wine_nn_config["BINARY_ACTIVATION"]
OPTIMIZER = wine_nn_config["OPTIMIZER"]
BINARY_LOSS = wine_nn_config["BINARY_LOSS"]
METRIC = wine_nn_config["METRIC"]

# Step 1: Load the wine quality dataset
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red_wine = pd.read_csv(red_wine_url, delimiter=";")
white_wine = pd.read_csv(white_wine_url, delimiter=";")

red_wine["type"] = 0
white_wine["type"] = 1

wine_data = pd.concat([red_wine, white_wine])
print(wine_data.head())

# Step 2: Process the wine dataset
# For red/white binary classification
X = wine_data.drop("quality", axis=1)
y = wine_data["quality"]

# For quality mul-ti classification
X = wine_data.drop(["quality", "type"], axis=1)
y = wine_data["type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# [Improvement] Multiple Scaler Options
# It is added to improve the performance of the NN, better converging speed
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Build and train the model
model = Sequential(
    [
        Dense(64, activation=ACTIVATION, input_shape=(X_train.shape[1],)),
        Dense(64, activation=ACTIVATION),
        Dense(1, activation=BINARY_ACTIVATION),
    ]
)

model.compile(optimizer=OPTIMIZER, loss=BINARY_LOSS, metrics=[METRIC])

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
)

for key in history.history.keys():
    print(f"{key}: {history.history[key]}")

# Step 4: Validate the model
test_loss, test_metric = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test {METRIC}: {test_metric}")

# Step 5: Make conclusions
# Plotting iterative learning curves by iteration
plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Train Loss", color="navy")
plt.plot(history.history["val_loss"], label="Validation Loss", color="lightcoral")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Iterative Learning Curve (Loss)")
plt.grid(visible=True)
plt.legend(frameon=False)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history["accuracy"], label="Train Accuracy", color="cornflowerblue")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="chartreuse")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Iterative Learning Curve (Accuracy)")
plt.grid(visible=True)
plt.legend(frameon=False)
plt.show()


# Step 6: Plot learning curve with varying training sizes
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    return plt


from sklearn.base import BaseEstimator, ClassifierMixin


class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(
            X,
            y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=0,
        )
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]


keras_clf = KerasClassifier(model)

plot_learning_curve(keras_clf, "Learning Curve", X_train, y_train, cv=3)
plt.show()
