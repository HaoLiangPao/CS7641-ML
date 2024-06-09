import json5
import os
import pandas as pd
import numpy as np
import tensorflow as tf

# Models
# NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier

# SVM
from sklearn.svm import SVC

# K-NN
from sklearn.neighbors import KNeighborsClassifier

# Boosting (Decision Tree)
from sklearn.ensemble import AdaBoostClassifier

# Model Tunning
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Plotting
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
import matplotlib.pyplot as plt

# Customized Libraries
from utils.plotting import (
    plot_learning_curve,
    plot_validation_curve,
    plot_iterative_learning_curves,
    plot_multiple_learning_curves,
    plot_learning_curve_with_test,
)

# Load configuration from JSON file
with open("../analysis_configs.json5", "r") as f:
    config = json5.load(f)

# Extract hyperparameters for Wine dataset NN
wine_nn_config = config["wine"]["NN"]
wine_svm_config = config["wine"]["SVM"]
wine_knn_config = config["wine"]["KNN"]
wine_boost_config = config["wine"]["BOOST"]

TEST_SIZE = wine_nn_config["TEST_SIZE"]
RANDOM_STATE = wine_nn_config["RANDOM_STATE"]
METRIC = wine_nn_config["METRIC"]

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
NN_HYPERPARAMETER_RANGES = wine_nn_config["HYPERPARAMETER_RANGES"]

# ========== Step 1: Load the wine quality dataset ==========
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red_wine = pd.read_csv(red_wine_url, delimiter=";")
white_wine = pd.read_csv(white_wine_url, delimiter=";")

red_wine["type"] = 0
white_wine["type"] = 1

wine_data = pd.concat([red_wine, white_wine])
print(wine_data.head())

# ========== Step 2: Process the wine dataset ==========
# For quality multi-class classification
X = wine_data.drop(["quality"], axis=1)
y = wine_data["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# [Improvement] Multiple Scaler Options (It is added to improve the performance of the NN, better converging speed)
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Plot validation curves


def create_nn_model(
    optimizer=OPTIMIZER,
    loss=MULTIPLE_LOSS,
    metrics=[METRIC],
    activation=ACTIVATION,
    output_activation=MULTIPLE_ACTIVATION,
):
    model = Sequential(
        [
            Dense(64, activation=activation, input_shape=(X_train.shape[1],)),
            Dense(64, activation=activation),
            Dense(10, activation=output_activation),  # 7 classes for labels 3 to 9
        ]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

nn_clf = KerasClassifier(
    model=create_nn_model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
)

for hyperparameter in NN_HYPERPARAMETER_RANGES:
    param_name = hyperparameter
    param_range = NN_HYPERPARAMETER_RANGES[hyperparameter]

    plot_validation_curve(
        nn_clf,
        f"NN Validation Curve ({param_name})",
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        save_path=f"images/nn_validation_curve_{param_name}.jpg",
    )

# LEARNING CURVE
# Load configuration from JSON file
with open("../best_configs.json5", "r") as f:
    config = json5.load(f)

# Extract hyperparameters for Wine dataset NN
wine_knn_config = config["wine"]["KNN"]

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

## [NN]
nn_model_best = create_nn_model(
    optimizer=OPTIMIZER,
    loss=MULTIPLE_LOSS,
    metrics=[METRIC],
    activation=ACTIVATION,
    output_activation=MULTIPLE_ACTIVATION,
)
nn_history_best = nn_model_best.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
)

nn_clf_best = KerasClassifier(
    model=nn_model_best,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
)

## [NN]
plot_learning_curve_with_test(
    nn_clf_best,
    "NN Learning Curve (Best Hyperparameters)",
    X_train,
    y_train,
    X_test,
    y_test,
    cv=3,
    save_path="images/nn_best_learning_curve_with_test.jpg",
)

plot_learning_curve(
    nn_clf_best,
    "NN Learning Curve (Best Hyperparameters)",
    X_train,
    y_train,
    cv=3,
    save_path="images/nn_best_learning_curve.jpg",
)

# Iterative learning curve
histories = [nn_history_best]
labels = ["Neural Network"]
plot_iterative_learning_curves(
    histories, labels, metric=METRIC, save_path="images/nn_best_iterative_learning_curve"
)

# Before tunning
nn_model = create_nn_model()
nn_history = nn_model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
)
nn_clf = KerasClassifier(
    model=nn_model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
)

plot_learning_curve_with_test(
    nn_clf,
    "NN Learning Curve",
    X_train,
    y_train,
    X_test,
    y_test,
    cv=3,
    save_path="images/nn_learning_curve_with_test.jpg",
)

plot_learning_curve(
    nn_clf,
    "NN Learning Curve",
    X_train,
    y_train,
    cv=3,
    save_path="images/nn_learning_curve.jpg",
)

# Iterative learning curve
histories = [nn_history_best,nn_history]
labels = ["Neural Network"]
plot_iterative_learning_curves(
    histories, labels, metric=METRIC, save_path="images/nn_iterative_learning_curve"
)
