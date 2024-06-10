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
from sklearn.tree import DecisionTreeClassifier
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

SVM_C = wine_svm_config["C"]
SVM_KERNEL = wine_svm_config["KERNEL"]
SVM_GAMMA = wine_svm_config["GAMMA"]
SVM_DEGREE = wine_svm_config["DEGREE"]  # For polynomial kernel
SVM_COEF0 = wine_svm_config["COEF0"]  # For polynomial and sigmoid kernels

KNN_N_NEIGHBORS = wine_knn_config["N_NEIGHBORS"]
KNN_WEIGHTS = wine_knn_config["WEIGHTS"]

BOOST_N_ESTIMATORS = wine_boost_config["N_ESTIMATORS"]
BOOST_LEARNING_RATE = wine_boost_config["LEARNING_RATE"]
BOOST_TREE_PARAMS = wine_boost_config["TREE_PARAMS"]

# ========== Step 1: Load the wine quality dataset ==========
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
data = pd.read_csv(
    url, header=None, names=columns, na_values=" ?", skipinitialspace=True
)

# Drop rows with missing values
data = data.dropna()

# Convert categorical columns to numerical
data = pd.get_dummies(data, drop_first=True)

# ========== Step 2: Process the wine dataset ==========
# Separate features and target
X = data.drop("income_>50K", axis=1)
y = data["income_>50K"]

# Split the data into train and test sets
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
# Initialize AdaBoostClassifier with a base decision tree classifier
boost_clf = AdaBoostClassifier(
    n_estimators=BOOST_N_ESTIMATORS[0],
    learning_rate=BOOST_LEARNING_RATE[0],
    random_state=RANDOM_STATE,
)
for param_name, param_range in zip(
    ["n_estimators", "learning_rate"], [BOOST_N_ESTIMATORS, BOOST_LEARNING_RATE]
):
    plot_validation_curve(
        boost_clf,
        f"Boosting Validation Curve ({param_name})",
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        save_path=f"images/boosting_validation_curve_{param_name}.jpg",
    )

boost_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier())
# Plot validation curves for tree pruning parameters
for param_name, param_range in BOOST_TREE_PARAMS.items():
    plot_validation_curve(
        boost_clf,
        f"Boosting Validation Curve ({param_name})",
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        save_path=f"images/boosting_validation_curve_{param_name}.jpg",
    )

# LEARNING CURVE
best_n_estimators = 100
best_learning_rate = 0.1

# Create new AdaBoostClassifier with the best parameters
best_boost_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=3, min_samples_split=5, min_samples_leaf=2
    ),
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate,
    random_state=RANDOM_STATE,
)

# Plot learning curve with test
plot_learning_curve_with_test(
    best_boost_clf,
    "Boosting Learning Curve (Best Hyperparameters)",
    X_train,
    y_train,
    X_test,
    y_test,
    cv=3,
    save_path="images/boosting_best_learning_curve_with_test.jpg",
)

plot_learning_curve(
    best_boost_clf,
    "SVM Learning Curve",
    X_train,
    y_train,
    cv=3,
    save_path="images/boosting_best_learning_curve.jpg",
)

# Before tunning
boost_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(),
    random_state=RANDOM_STATE,
)
# Plot learning curve with test
plot_learning_curve_with_test(
    boost_clf,
    "Boosting Learning Curve",
    X_train,
    y_train,
    X_test,
    y_test,
    cv=3,
    save_path="images/boosting_learning_curve_with_test.jpg",
)

plot_learning_curve(
    boost_clf,
    "SVM Learning Curve",
    X_train,
    y_train,
    cv=3,
    save_path="images/boosting_learning_curve.jpg",
)
