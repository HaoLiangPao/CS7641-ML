import subprocess
import sys

sys.path.append("../pyperch")

import numpy as np
from sklearn.datasets import make_classification
from torch import nn, optim
from skorch import NeuralNetClassifier
from pyperch.neural.rhc_nn import RHCModule
import matplotlib.pyplot as plt

X, y = make_classification(1000, 12, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)

from skorch.callbacks import EpochScoring

net = NeuralNetClassifier(
    module=RHCModule,
    module__input_dim=12,
    module__output_dim=2,
    module__hidden_units=20,
    module__hidden_layers=1,
    module__step_size=0.05,
    module__dropout_percent=0,
    module__activation=nn.ReLU(),
    module__output_activation=nn.Softmax(dim=-1),
    max_epochs=5000,
    verbose=0,
    callbacks=[
        EpochScoring(scoring="accuracy", name="train_acc", on_train=True),
    ],
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
)

RHCModule.register_rhc_training_step()

# fit data
net.fit(X, y)

plt.figure()
# plot the iterative learning curve (loss)
plt.plot(net.history[:, "train_loss"], label="Train Loss", color="navy")
plt.plot(net.history[:, "valid_loss"], label="Validation Loss", color="lightcoral")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Iterative Learning Curve (Loss)")
plt.grid(visible=True)
plt.legend(frameon=False)
plt.show()

plt.figure()
# plot the iterative learning curve (accuracy)
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
# Plot the learning curve
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

gs.fit(X, y)
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
