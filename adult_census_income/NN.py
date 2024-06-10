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


def create_nn_model(
    optimizer=OPTIMIZER,
    loss=MULTIPLE_LOSS,
    metrics=[METRIC],
    activation=ACTIVATION,
    output_activation=MULTIPLE_ACTIVATION,
    learning_rate=0.001,
    num_layers=1,
    units_per_layer=64,
):
    if optimizer == "adam":
        opt = Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model = Sequential()
    model.add(
        Dense(units_per_layer, activation=activation, input_shape=(X_train.shape[1],))
    )

    for _ in range(num_layers - 1):
        model.add(Dense(units_per_layer, activation=activation))

    model.add(
        Dense(10, activation=output_activation)
    )  # Adjust the output layer for the number of classes
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
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

    if param_name in ["learning_rate", "num_layers", "units_per_layer"]:
        plot_validation_curve(
            KerasClassifier(
                model=create_nn_model,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
            ),
            f"NN Validation Curve ({param_name})",
            X_train,
            y_train,
            param_name=f"model__{param_name}",
            param_range=param_range,
            cv=3,
            save_path=f"images/nn_validation_curve_{param_name}.jpg",
        )
    else:
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

LEARNING_RATE = wine_nn_config["learning_rate"]
NUM_LAYERS = wine_nn_config["num_layers"]
UNITES_PER_LAYER = wine_nn_config["units_per_layer"]

## [NN]
nn_model_best = create_nn_model(
    optimizer=OPTIMIZER,
    loss=MULTIPLE_LOSS,
    metrics=[METRIC],
    activation=ACTIVATION,
    output_activation=MULTIPLE_ACTIVATION,
    learning_rate=LEARNING_RATE,
    num_layers=NUM_LAYERS,
    units_per_layer=UNITES_PER_LAYER
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
    histories,
    labels,
    metric=METRIC,
    save_path="images/nn_best_iterative_learning_curve",
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
histories = [nn_history_best, nn_history]
labels = ["Neural Network"]
plot_iterative_learning_curves(
    histories, labels, metric=METRIC, save_path="images/nn_iterative_learning_curve"
)
