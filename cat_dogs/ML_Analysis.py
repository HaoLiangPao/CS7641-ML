import json5
import os
import pandas as pd
import numpy as np
import tensorflow as tf

# Models
# NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier

# SVM
from sklearn.svm import SVC

# K-NN
from sklearn.neighbors import KNeighborsClassifier

# Boosting (Decision Tree)
from sklearn.ensemble import AdaBoostClassifier

# Model Tuning
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Plotting
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
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

# Extract hyperparameters for FASHION_MNIST dataset NN
fashion_mnist_nn_config = config["fashion_mnist"]["NN"]
fashion_mnist_svm_config = config["fashion_mnist"]["SVM"]
fashion_mnist_knn_config = config["fashion_mnist"]["KNN"]
fashion_mnist_boost_config = config["fashion_mnist"]["BOOST"]

TEST_SIZE = fashion_mnist_nn_config["TEST_SIZE"]
RANDOM_STATE = fashion_mnist_nn_config["RANDOM_STATE"]
EPOCHS = fashion_mnist_nn_config["EPOCHS"]
BATCH_SIZE = fashion_mnist_nn_config["BATCH_SIZE"]
VALIDATION_SPLIT = fashion_mnist_nn_config["VALIDATION_SPLIT"]
ACTIVATION = fashion_mnist_nn_config["ACTIVATION"]
OPTIMIZER = fashion_mnist_nn_config["OPTIMIZER"]
LOSS = fashion_mnist_nn_config["LOSS"]
METRIC = fashion_mnist_nn_config["METRIC"]
NN_HYPERPARAMETER_RANGES = fashion_mnist_nn_config["HYPERPARAMETER_RANGES"]

SVM_C = fashion_mnist_svm_config["C"]
SVM_KERNEL = fashion_mnist_svm_config["KERNEL"]
SVM_GAMMA = fashion_mnist_svm_config["GAMMA"]
SVM_DEGREE = fashion_mnist_svm_config["DEGREE"]  # For polynomial kernel
SVM_COEF0 = fashion_mnist_svm_config["COEF0"]  # For polynomial and sigmoid kernels

KNN_N_NEIGHBORS = fashion_mnist_knn_config["N_NEIGHBORS"]
KNN_WEIGHTS = fashion_mnist_knn_config["WEIGHTS"]

BOOST_N_ESTIMATORS = fashion_mnist_boost_config["N_ESTIMATORS"]
BOOST_LEARNING_RATE = fashion_mnist_boost_config["LEARNING_RATE"]

# ========== Step 1: Load the FASHION_MNIST dataset ==========
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten the images for non-NN models
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# ========== Step 2: Process the FASHION_MNIST dataset ==========
# [Improvement] Multiple Scaler Options (It is added to improve the performance of the NN, better converging speed)
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()

X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)


# ========== Step 3: Build and train the models ==========
# NN Model
def create_nn_model(
    optimizer=OPTIMIZER,
    loss=LOSS,
    metrics=[METRIC],
    activation=ACTIVATION,
):
    model = Sequential(
        [
            Flatten(input_shape=(28, 28)),
            Dense(128, activation=activation),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


nn_model = create_nn_model()
nn_history = nn_model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
)

# SVM Model
svm_clf = SVC(kernel=SVM_KERNEL, random_state=RANDOM_STATE, probability=True)
svm_clf.fit(X_train_flat, y_train)

# KNN Model
knn_clf = KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS[0], weights=KNN_WEIGHTS[0])
knn_clf.fit(X_train_flat, y_train)

# Boosting Model
boost_clf = AdaBoostClassifier(
    n_estimators=BOOST_N_ESTIMATORS[0],
    learning_rate=BOOST_LEARNING_RATE[0],
    random_state=RANDOM_STATE,
)
boost_clf.fit(X_train_flat, y_train)

# ========== Step 4: Validate the models ==========
# NN
nn_test_loss, nn_test_metric = nn_model.evaluate(X_test, y_test)
print(f"NN Test Loss: {nn_test_loss}")
print(f"NN Test {METRIC}: {nn_test_metric}")

# SVM
svm_test_metric = svm_clf.score(X_test_flat, y_test)
print(f"SVM Test {METRIC}: {svm_test_metric}")

# KNN
knn_test_metric = knn_clf.score(X_test_flat, y_test)
print(f"KNN Test {METRIC}: {knn_test_metric}")

# Boosting
boost_test_metric = boost_clf.score(X_test_flat, y_test)
print(f"Boosting Test {METRIC}: {boost_test_metric}")

# ========== Step 5: Plot learning curves ==========
# NN
nn_clf = KerasClassifier(
    model=create_nn_model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
)

plot_learning_curve(
    nn_clf,
    "NN Learning Curve",
    X_train_flat,
    y_train,
    cv=3,
    save_path="images/fashion_mnist_nn_learning_curve.jpg",
)

# SVM
plot_learning_curve(
    svm_clf,
    "SVM Learning Curve",
    X_train_flat,
    y_train,
    cv=3,
    save_path="images/fashion_mnist_svm_learning_curve.jpg",
)

# KNN
plot_learning_curve(
    knn_clf,
    "KNN Learning Curve",
    X_train_flat,
    y_train,
    cv=3,
    save_path="images/fashion_mnist_knn_learning_curve.jpg",
)

# Boosting
plot_learning_curve(
    boost_clf,
    "Boosting Learning Curve",
    X_train_flat,
    y_train,
    cv=3,
    save_path="images/fashion_mnist_boosting_learning_curve.jpg",
)

# ========== Step 6: Plot validation curves ==========
# NN Validation Curves
for hyperparameter in NN_HYPERPARAMETER_RANGES:
    param_name = hyperparameter
    param_range = NN_HYPERPARAMETER_RANGES[hyperparameter]

    plot_validation_curve(
        nn_clf,
        f"NN Validation Curve ({param_name})",
        X_train_flat,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        save_path=f"images/fashion_mnist_nn_validation_curve_{param_name}.jpg",
    )

# SVM
for param_name, param_range in zip(
    ["C", "gamma", "degree", "coef0"], [SVM_C, SVM_GAMMA, SVM_DEGREE, SVM_COEF0]
):
    plot_validation_curve(
        svm_clf,
        f"SVM Validation Curve ({param_name})",
        X_train_flat,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        save_path=f"images/fashion_mnist_svm_validation_curve_{param_name}.jpg",
    )

# KNN
for param_name, param_range in zip(
    ["n_neighbors", "weights"], [KNN_N_NEIGHBORS, KNN_WEIGHTS]
):
    plot_validation_curve(
        knn_clf,
        f"KNN Validation Curve ({param_name})",
        X_train_flat,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        save_path=f"images/fashion_mnist_knn_validation_curve_{param_name}.jpg",
    )

# Boosting
for param_name, param_range in zip(
    ["n_estimators", "learning_rate"], [BOOST_N_ESTIMATORS, BOOST_LEARNING_RATE]
):
    plot_validation_curve(
        boost_clf,
        f"Boosting Validation Curve ({param_name})",
        X_train_flat,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        save_path=f"images/fashion_mnist_boosting_validation_curve_{param_name}.jpg",
    )

# ========== Step 8: Plot combined iterative learning curves ==========
histories = [nn_history]
labels = ["Neural Network"]
plot_iterative_learning_curves(
    histories,
    labels,
    metric=METRIC,
    save_path="images/fashion_mnist_iterative_learning_curve",
)

# SVM Comparison between different kernel functions
kernel_functions = ["linear", "poly", "rbf", "sigmoid"]
histories = []
labels = []

for kernel in kernel_functions:
    svm_clf = SVC(kernel=kernel, random_state=RANDOM_STATE, C=1, gamma="scale")
    svm_clf.fit(X_train_flat, y_train)

    # Plot learning curve
    plot_learning_curve(
        svm_clf,
        f"SVM Learning Curve ({kernel} kernel)",
        X_train_flat,
        y_train,
        cv=3,
        save_path=f"images/fashion_mnist_svm_learning_curve_{kernel}.jpg",
    )

    # Save history and labels for combined iterative learning curve
    histories.append(svm_clf)
    labels.append(f"SVM ({kernel})")
