import subprocess
import sys

sys.path.append("../pyperch")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from skorch import NeuralNetClassifier
from pyperch.neural.sa_nn import SAModule
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

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]
output_dim = len(np.unique(y))


from skorch.callbacks import EpochScoring

# Define the neural network with backpropagation
net = NeuralNetClassifier(
    module=SAModule,
    module__input_dim=input_dim,  # Number of features
    module__output_dim=output_dim,  # Number of classes
    module__hidden_units=32,  # Number of hidden units
    module__hidden_layers=2,  # Number of hidden layers
    module__step_size=0.1,
    module__t=20000,
    module__cooling=0.99,
    # module__dropout_percent=0,
    # module__activation=nn.ReLU(),
    # module__output_activation=nn.Softmax(dim=-1),
    max_epochs=500,
    verbose=0,
    callbacks=[EpochScoring(scoring="accuracy", name="train_acc", on_train=True)],
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.SGD,
    lr=0.05,
    iterator_train__shuffle=True,
)

SAModule.register_sa_training_step()

# Train the network
net.fit(X_train, y_train)

# Predict class probabilities
y_proba = net.predict_proba(X_train)

# Plot iterative learning curve (loss)
plt.figure()
plt.plot(net.history[:, "train_loss"], label="Train Loss", color="navy")
plt.plot(net.history[:, "valid_loss"], label="Validation Loss", color="lightcoral")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Iterative Learning Curve (Loss)")
plt.grid(visible=True)
plt.legend(frameon=False)
plt.show()

# Plot iterative learning curve (accuracy)
plt.figure()
plt.plot(net.history[:, "train_acc"], label="Train Acc", color="cornflowerblue")
plt.plot(net.history[:, "valid_acc"], label="Validation Acc", color="chartreuse")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Iterative Learning Curve (Accuracy)")
plt.grid(visible=True)
plt.legend(frameon=False)
plt.show()

from sklearn.model_selection import learning_curve

# Plot the learning curve
plt.figure()

train_sizes, train_scores, test_scores = learning_curve(
    net,
    X_train,
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3,
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
plt.title("Learning Curve")
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

# Deactivate skorch-internal train-valid split and verbose logging
net.set_params(train_split=False, verbose=0)

default_params = {
    "module__input_dim": [X.shape[1]],
    "module__output_dim": [len(np.unique(y))],
}

# Module-specific parameters need to begin with 'module__'
params = {
    "lr": [0.01, 0.02],
    "max_epochs": [10, 20],
    "module__hidden_units": [10, 20],
    **default_params,
}

# Fit GridSearchCV on the training data
gs = GridSearchCV(net, params, refit=False, cv=3, scoring="accuracy", verbose=2)
gs.fit(X_train, y_train)
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
