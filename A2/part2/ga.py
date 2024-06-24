import subprocess
import sys

sys.path.append("../pyperch")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from skorch import NeuralNetClassifier
from pyperch.neural.ga_nn import GAModule
import matplotlib.pyplot as plt

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

# # Drop rows with missing values
data = data.dropna()

# Convert categorical columns to numerical
data = pd.get_dummies(data, drop_first=True)

# ========== Step 2: Process the wine dataset ==========
# Separate features and target
X = data.drop("income_>50K", axis=1)
y = data["income_>50K"]

X = X.astype(np.float32)
y = y.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=42
)

# X_train = X_train.astype(np.float32)
# y_train = y_train.astype(np.int64)

# X_test = X_test.astype(np.float32)
# y_test = y_test.astype(np.int64)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]
output_dim = len(np.unique(y))

from skorch.callbacks import EpochScoring

net = NeuralNetClassifier(
    module=GAModule,
    module__input_dim=input_dim,
    module__output_dim=output_dim,
    module__hidden_units=32,
    module__hidden_layers=2,
    module__population_size=300,
    module__to_mate=150,
    module__to_mutate=30,
    module__dropout_percent=0,
    module__step_size=0.1,
    module__activation=nn.ReLU(),
    module__output_activation=nn.Softmax(dim=-1),
    max_epochs=100,
    verbose=0,
    callbacks=[
        EpochScoring(scoring="accuracy", name="train_acc", on_train=True),
    ],
    # use nn.CrossEntropyLoss instead of default nn.NLLLoss
    # for use with raw prediction values instead of log probabilities
    criterion=nn.CrossEntropyLoss(),
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)

GAModule.register_ga_training_step()

# fit data
net.fit(X_train, y_train)

plt.figure()
# plot the iterative learning curve (loss)
plt.plot(net.history[:, "train_loss"], label="Train Loss", color="navy")
plt.plot(net.history[:, "valid_loss"], label="Validation Loss", color="lightcoral")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Iterative Learning Curve - GA (Loss)")
plt.grid(visible=True)
plt.legend(frameon=False)
plt.show()

plt.figure()
# plot the iterative learning curve (accuracy)
plt.plot(net.history[:, "train_acc"], label="Train Acc", color="cornflowerblue")
plt.plot(net.history[:, "valid_acc"], label="Validation Acc", color="chartreuse")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Iterative Learning Curve - GA (Accuracy)")
plt.grid(visible=True)
plt.legend(frameon=False)
plt.show()

from sklearn.model_selection import learning_curve

plt.figure()
# Plot the learning curve
train_sizes, train_scores, test_scores = learning_curve(
    net, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 5), cv=3
)

train_scores_mean = train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
test_scores_mean = test_scores.mean(axis=1)
test_scores_std = test_scores.std(axis=1)
plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="cyan",
)
plt.fill_between(
    train_sizes,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="darkorchid",
)
plt.plot(train_sizes, train_scores_mean, label="Training score", color="cyan")
plt.plot(train_sizes, test_scores_mean, label="Test score", color="darkorchid")
plt.title("Learning Curve - GA")
plt.xlabel("Training size")
plt.ylabel("Score")
plt.grid(visible=True)
plt.legend(frameon=False)
plt.show()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline(
    [
        ("scale", StandardScaler()),
        ("net", net),
    ]
)

pipe.fit(X, y)
y_proba = pipe.predict_proba(X)

from sklearn.model_selection import GridSearchCV

# deactivate skorch-internal train-valid split and verbose logging
net.set_params(
    train_split=False,
    verbose=0,
)

# module specific parameters need to begin with 'module__'
default_params = {
    "module__input_dim": [12],
    "module__output_dim": [2],
    "module__step_size": [0.1],
}

grid_search_params = {
    "max_epochs": [10, 20],
    "module__hidden_units": [10, 20],
    "module__hidden_layers": [1, 2],
    "module__activation": [nn.ReLU(), nn.Tanh()],
    **default_params,
}

gs = GridSearchCV(
    net, grid_search_params, refit=False, cv=3, scoring="accuracy", verbose=2
)

gs.fit(X_train, y_train)
print(
    "[Fitting trainning sets] best score: {:.3f}, best params: {}".format(
        gs.best_score_, gs.best_params_
    )
)

gs.fit(X_test, y_test)
print(
    "[Fitting testing sets] best score: {:.3f}, best params: {}".format(
        gs.best_score_, gs.best_params_
    )
)
