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

KNN_N_NEIGHBORS = wine_knn_config["N_NEIGHBORS"]
KNN_WEIGHTS = wine_knn_config["WEIGHTS"]

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
# [KNN]
for param_name, param_range in zip(
    ["n_neighbors", "weights"], [KNN_N_NEIGHBORS, KNN_WEIGHTS]
):
    plot_validation_curve(
        KNeighborsClassifier(),
        f"KNN Validation Curve ({param_name})",
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        save_path=f"images/knn_validation_curve_{param_name}.jpg",
    )

# LEARNING CURVE

# Load configuration from JSON file
with open("../best_configs.json5", "r") as f:
    config = json5.load(f)

# Extract hyperparameters for Wine dataset NN
wine_knn_config = config["wine"]["KNN"]

KNN_N_NEIGHBORS = wine_knn_config["N_NEIGHBORS"]
KNN_WEIGHTS = wine_knn_config["WEIGHTS"]

## [KNN]
knn_best = KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS)
knn_best.fit(X_train, y_train)

knn_test_metric = knn_best.score(X_test, y_test)
print(f"KNN Test {METRIC}: {knn_test_metric}")

## [KNN]
plot_learning_curve_with_test(
    knn_best,
    "KNN Learning Curve (Best Hyperparameters)",
    X_train,
    y_train,
    X_test,
    y_test,
    cv=3,
    save_path="images/knn_best_learning_curve_with_test.jpg",
)

plot_learning_curve(
    knn_best,
    "KNN Learning Curve (Best Hyperparameters)",
    X_train,
    y_train,
    cv=3,
    save_path="images/knn_best_learning_curve.jpg",
)

# Before tunning
knn_clf = KNeighborsClassifier(n_neighbors=3, weights="uniform")
knn_clf.fit(X_train, y_train)
plot_learning_curve_with_test(
    knn_clf,
    "KNN Learning Curve",
    X_train,
    y_train,
    X_test,
    y_test,
    cv=3,
    save_path="images/knn_learning_curve_with_test.jpg",
)

plot_learning_curve(
    knn_clf,
    "KNN Learning Curve",
    X_train,
    y_train,
    cv=3,
    save_path="images/knn_learning_curve.jpg",
)
