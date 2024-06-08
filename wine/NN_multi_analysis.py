import json5
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
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
wine_svm_config = config["wine"]["SVM"]

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
NN_HYPERPARAMETER_RANGES = wine_nn_config["HYPERPARAMETER_RANGES"]

SVM_C = wine_svm_config["C"]
SVM_KERNEL = wine_svm_config["KERNEL"]
SVM_GAMMA = wine_svm_config["GAMMA"]

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


# ========== Step 3: Build and train the NN model ==========

##[NN]

def create_nn_model(
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

## [SVM]
svm_clf = SVC(kernel=SVM_KERNEL, random_state=RANDOM_STATE, probability=True)
svm_history = svm_clf.fit(X_train, y_train)

# ========== Step 4: Validate the model ==========
## [NN]
test_loss, test_metric = nn_model.evaluate(X_test, y_test)
print(f"NN Test Loss: {test_loss}")
print(f"NN Test {METRIC}: {test_metric}")

## [SVM]

svm_test_metric = svm_clf.score(X_test, y_test)
print(f"SVM Test {METRIC}: {svm_test_metric}")


# ========== Step 5: Plot learning curves ==========
## [NN]
nn_clf = KerasClassifier(
    model=create_nn_model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
)

plot_learning_curve(
    nn_clf,
    "NN Learning Curve",
    X_train,
    y_train,
    cv=3,
    save_path="images/nn_learning_curve.jpg",
)

## [SVM]
plot_learning_curve(
    svm_clf,
    "SVM Learning Curve",
    X_train,
    y_train,
    cv=3,
    save_path="images/svm_learning_curve.jpg",
)

# ========== Step 6: Plot validation curves ==========
# NN Validation Curves
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

# SVM Validation Curves
for param_name, param_range in zip(["C", "gamma"], [SVM_C, SVM_GAMMA]):
    plot_validation_curve(
        svm_clf,
        f"SVM Validation Curve ({param_name})",
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        cv=3,
        save_path=f"images/svm_validation_curve_{param_name}.jpg",
    )

# ========== Step 8: Plot combined iterative learning curves ==========
# histories = [nn_history, svm_history]
# labels = ["Neural Network", "SVM"]
histories = [nn_history]
labels = ["Neural Network"]
plot_iterative_learning_curves(
    histories, labels, metric=METRIC, save_path="images/iterative_learning_curve"
)


## [SVM] Comparision between different kernal functions
kernel_functions = ["linear", "poly", "rbf", "sigmoid"]
histories = []
labels = []

for kernel in kernel_functions:
    svm_clf = SVC(
        kernel=kernel, random_state=RANDOM_STATE, C=1, gamma="scale"
    )  # Fixed C and gamma for comparison
    svm_clf.fit(X_train, y_train)

    # Plot learning curve
    plot_learning_curve(
        svm_clf,
        f"SVM Learning Curve ({kernel} kernel)",
        X_train,
        y_train,
        cv=3,
        save_path=f"images/svm_learning_curve_{kernel}.jpg",
    )

    # Save history and labels for combined iterative learning curve
    histories.append(svm_clf)
    labels.append(f"SVM ({kernel})")