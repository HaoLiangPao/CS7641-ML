import subprocess
import sys

sys.path.append("../pyperch")

import numpy as np
from sklearn.datasets import make_classification
from torch import nn, optim
from skorch import NeuralNetClassifier
from pyperch.neural.backprop_nn import BackpropModule
import matplotlib.pyplot as plt

X, y = make_classification(1000, 12, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)

from skorch.callbacks import EpochScoring

net = NeuralNetClassifier(
    module=BackpropModule,
    module__input_dim=12,
    module__output_dim=2,
    module__hidden_units=30,
    module__hidden_layers=1,
    max_epochs=500,
    verbose=0,
    callbacks=[
        EpochScoring(scoring="accuracy", name="train_acc", on_train=True),
    ],
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.SGD,
    lr=0.05,
    iterator_train__shuffle=True,
)

net.fit(X, y)
y_proba = net.predict_proba(X)

plt.figure()
plt.plot(net.history[:, "train_loss"], label="Train Loss", color="navy")
plt.plot(net.history[:, "valid_loss"], label="Validation Loss", color="lightcoral")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Iterative Learning Curve (Loss)")
plt.grid(visible=True)
plt.legend(frameon=False)
plt.show()

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

plt.figure()

train_sizes, train_scores, test_scores = learning_curve(
    net, X, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=3
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

net.set_params(train_split=False, verbose=0)

default_params = {
    "module__input_dim": [12],
    "module__output_dim": [2],
}

params = {
    "lr": [0.01, 0.02],
    "max_epochs": [10, 20],
    "module__hidden_units": [10, 20],
    **default_params,
}
gs = GridSearchCV(net, params, refit=False, cv=3, scoring="accuracy", verbose=2)

gs.fit(X, y)
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
