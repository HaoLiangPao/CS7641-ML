import json5
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

# Customized Libraries
from utils.plotting import (
    plot_learning_curve,
    plot_validation_curve,
    plot_iterative_learning_curves,
)

# Load configuration from JSON file
with open("../analysis_configs.json5", "r") as f:
    config = json5.load(f)

# Extract hyperparameters for Wine dataset NN
wine_nn_config = config["wine"]["NN"]

TEST_SIZE = wine_nn_config["TEST_SIZE"]
RANDOM_STATE = wine_nn_config["RANDOM_STATE"]
EPOCHS = wine_nn_config["EPOCHS"]
BATCH_SIZE = wine_nn_config["BATCH_SIZE"]
VALIDATION_SPLIT = wine_nn_config["VALIDATION_SPLIT"]
ACTIVATION = wine_nn_config["ACTIVATION"]
BINARY_ACTIVATION = wine_nn_config["BINARY_ACTIVATION"]
MULTIPLE_ACTIVATION = wine_nn_config["MULTIPLE_ACTIVATION"]
OPTIMIZER = wine_nn_config["OPTIMIZER"]
BINARY_LOSS = wine_nn_config["BINARY_LOSS"]
MULTIPLE_LOSS = wine_nn_config["MULTIPLE_LOSS"]
METRIC = wine_nn_config["METRIC"]
HYPERPARAMETER_RANGES = wine_nn_config["HYPERPARAMETER_RANGES"]

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
# # For red/white binary classification
# X = wine_data.drop("type", axis=1)
# y = wine_data["type"]

# For quality mul-ti classification
X = wine_data.drop(["quality"], axis=1)
y = wine_data["quality"]

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

# binary model (Q1: red/white)
# model = Sequential(
#     [
#         Dense(64, activation=ACTIVATION, input_shape=(X_train.shape[1],)),
#         Dense(64, activation=ACTIVATION),
#         Dense(1, activation=BINARY_ACTIVATION),
#     ]
# )

# model.compile(optimizer=OPTIMIZER, loss=BINARY_LOSS, metrics=[METRIC])

# history = model.fit(
#     X_train,
#     y_train,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     validation_split=VALIDATION_SPLIT,
# )

# for key in history.history.keys():
#     print(f"{key}: {history.history[key]}")

# multiple model (Q2: wine quality)


# Function to create model
def create_model(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    activation="relu",
    output_activation="softmax",
):
    model = Sequential(
        [
            Dense(64, activation=activation, input_shape=(X_train.shape[1],)),
            Dense(64, activation=activation),
            Dense(10, activation=output_activation),  # 7 classes for labels 3 to 9
            # Dense(7, activation=output_activation),  # 7 classes for labels 3 to 9
        ]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

model = create_model()

model.compile(optimizer=OPTIMIZER, loss=MULTIPLE_LOSS, metrics=[METRIC])

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
plot_iterative_learning_curves(
    history, metric=METRIC, save_path="images/nn_iterative_learning_curve"
)

# Step 6: Plot learning curve with varying training sizes
keras_clf = KerasClassifier(
    model=create_model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
)
# 1. Plot learning curve
plot_learning_curve(
    keras_clf, 
    "Learning Curve", 
    X_train, 
    y_train, 
    cv=3,
    save_path="images/nn_learning_curve.jpg"
)
plt.show()

# 2. Plot validation curve with parameter range for 'param_name'
for hyperparameter in HYPERPARAMETER_RANGES:
    param_name = hyperparameter
    param_range = HYPERPARAMETER_RANGES[hyperparameter]

    plot_validation_curve(
        keras_clf,
        f"Validation Curve ({param_name})",
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        save_path=f"images/nn_validation_curve_{param_name}.jpg",
    )
    plt.show()
