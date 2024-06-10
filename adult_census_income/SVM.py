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

# SVM Model
svm_clf_rbf = SVC(kernel="rbf", random_state=RANDOM_STATE)
svm_clf_poly = SVC(kernel="poly", random_state=RANDOM_STATE)
svm_clf_sigmoid = SVC(kernel="sigmoid", random_state=RANDOM_STATE)
svm_clf_linear = SVC(kernel="linear", random_state=RANDOM_STATE)

# 1. Plot combined learning curves for different SVM kernels
svm_estimators = [svm_clf_linear, svm_clf_poly, svm_clf_rbf, svm_clf_sigmoid]
labels = ["Linear Kernel", "Polynomial Kernel", "RBF Kernel", "Sigmoid Kernel"]

plot_multiple_learning_curves(
    svm_estimators,
    labels,
    "SVM Learning Curves for Different Kernels",
    X_train,
    y_train,
    cv=3,
    save_path="images/svm_combined_learning_curves.jpg",
)

# 2. Plot validation curves
# Validation curve for C
plot_validation_curve(
    SVC(kernel="rbf"),
    title="Validation Curve for SVM (C)",
    X=X_train,
    y=y_train,
    param_name="C",
    param_range=SVM_C,
    cv=3,
    save_path="images/svm_validation_curve_C.jpg",
)

# Validation curve for gamma
plot_validation_curve(
    SVC(kernel="rbf"),
    title="Validation Curve for SVM (gamma)",
    X=X_train,
    y=y_train,
    param_name="gamma",
    param_range=SVM_GAMMA,
    cv=3,
    save_path="images/svm_validation_curve_gamma.jpg",
)

# If using polynomial kernel, also consider degree and coef0
plot_validation_curve(
    SVC(kernel="poly"),
    title="Validation Curve for SVM (degree)",
    X=X_train,
    y=y_train,
    param_name="degree",
    param_range=SVM_DEGREE,
    cv=3,
    save_path="images/svm_validation_curve_degree.jpg",
)

plot_validation_curve(
    SVC(kernel="poly"),
    title="Validation Curve for SVM (coef0)",
    X=X_train,
    y=y_train,
    param_name="coef0",
    param_range=SVM_COEF0,
    cv=3,
    save_path="images/svm_validation_curve_coef0.jpg",
)


# LEARNING CURVE

## [SVM]
svm_best = SVC(
    kernel="rbf", C=100, gamma=1, random_state=RANDOM_STATE, probability=True
)
svm_best.fit(X_train, y_train)

svm_test_metric = svm_best.score(X_test, y_test)
print(f"SVM Test {METRIC}: {svm_test_metric}")

## [SVM]
plot_learning_curve_with_test(
    svm_best,
    "SVM Learning Curve (Best Hyperparameters)",
    X_train,
    y_train,
    X_test,
    y_test,
    cv=3,
    save_path="images/svm_best_learning_curve_with_test.jpg",
)

plot_learning_curve(
    svm_best,
    "SVM Learning Curve",
    X_train,
    y_train,
    cv=3,
    save_path="images/svm_best_learning_curve.jpg",
)

# Before tunning
svm_clf = SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True)
svm_clf.fit(X_train, y_train)

svm_test_metric = svm_clf.score(X_test, y_test)
print(f"SVM Test {METRIC}: {svm_test_metric}")


## [SVM]
plot_learning_curve_with_test(
    svm_clf,
    "SVM Learning Curve (Best Hyperparameters)",
    X_train,
    y_train,
    X_test,
    y_test,
    cv=3,
    save_path="images/svm_clf_learning_curve_with_test.jpg",
)

plot_learning_curve(
    svm_clf,
    "SVM Learning Curve",
    X_train,
    y_train,
    cv=3,
    save_path="images/svm_clf_learning_curve.jpg",
)
