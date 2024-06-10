import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    save_path=None,
    custom_params=None,
):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if custom_params:
        for key, value in custom_params.items():
            plt.setp(plt.gca(), key, value)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_validation_curve(
    estimator,
    title,
    X,
    y,
    param_name,
    param_range,
    ylim=None,
    cv=None,
    n_jobs=None,
    save_path=None,
    custom_params=None,
):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(param_name)
    plt.ylabel("Score")

    if custom_params:
        for key, value in custom_params.items():
            plt.setp(plt.gca(), key, value)

    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        n_jobs=n_jobs,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(param_range, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        param_range, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_iterative_learning_curves(history, metric="accuracy", save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"], label="Train Loss", color="navy")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="lightcoral")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Iterative Learning Curve (Loss)")
    plt.grid(visible=True)
    plt.legend(frameon=False)
    if save_path:
        plt.savefig(save_path + "_loss.jpg")
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(history.history[metric], label="Train Accuracy", color="cornflowerblue")
    plt.plot(
        history.history[f"val_{metric}"],
        label="Validation Accuracy",
        color="chartreuse",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Iterative Learning Curve (Accuracy)")
    plt.grid(visible=True)
    plt.legend(frameon=False)
    if save_path:
        plt.savefig(save_path + "_accuracy.jpg")
    else:
        plt.show()
    plt.close()


def plot_iterative_learning_curves(
    histories, labels, metric="accuracy", save_path=None
):
    plt.figure(figsize=(12, 6))
    for history, label in zip(histories, labels):
        plt.plot(history.history["loss"], label=f"{label} Train Loss")
        plt.plot(history.history["val_loss"], label=f"{label} Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Iterative Learning Curve (Loss)")
    plt.grid(visible=True)
    plt.legend(frameon=False)
    if save_path:
        plt.savefig(save_path + "_loss.jpg")
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    for history, label in zip(histories, labels):
        plt.plot(history.history[metric], label=f"{label} Train {metric.capitalize()}")
        plt.plot(
            history.history[f"val_{metric}"],
            label=f"{label} Validation {metric.capitalize()}",
        )
    plt.xlabel("Iteration")
    plt.ylabel(metric.capitalize())
    plt.title(f"Iterative Learning Curve ({metric.capitalize()})")
    plt.grid(visible=True)
    plt.legend(frameon=False)
    if save_path:
        plt.savefig(save_path + f"_{metric}.jpg")
    else:
        plt.show()
    plt.close()


# Function to plot multiple learning curves
def plot_multiple_learning_curves(
    estimators,
    labels,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    save_path=None,
    custom_params=None,
):
    # Used for SVM to compare different kernel functions
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if custom_params:
        for key, value in custom_params.items():
            plt.setp(plt.gca(), key, value)

    colors = matplotlib.pyplot.get_cmap("tab10", len(estimators)).colors

    for idx, (estimator, label) in enumerate(zip(estimators, labels)):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        color = colors[idx]

        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color=color,
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color=color,
        )
        plt.plot(
            train_sizes,
            train_scores_mean,
            "o-",
            color=color,
            label=f"{label} Training score",
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o--",
            color=color,
            label=f"{label} Validation score",
        )

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_learning_curve_with_test(
    estimator,
    title,
    X_train,
    y_train,
    X_test,
    y_test,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    save_path=None,
    custom_params=None,
):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if custom_params:
        for key, value in custom_params.items():
            plt.setp(plt.gca(), key, value)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    # Calculate test scores for different training sizes
    test_scores = []
    for size in train_sizes:
        X_train_subset = X_train[: int(size)]
        y_train_subset = y_train[: int(size)]
        estimator.fit(X_train_subset, y_train_subset)
        test_scores.append(estimator.score(X_test, y_test))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    test_scores = np.array(test_scores)

    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, "o--", color="g", label="Validation score")
    plt.plot(train_sizes, test_scores, "o-", color="b", label="Test score")

    plt.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


from sklearn.metrics import accuracy_score


def plot_learning_curve_with_test(
    estimator,
    title,
    X_train,
    y_train,
    X_test,
    y_test,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    save_path=None,
    custom_params=None,
):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if custom_params:
        for key, value in custom_params.items():
            plt.setp(plt.gca(), key, value)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    # Compute test scores manually
    test_scores_manual = []
    for train_size in train_sizes:
        X_train_subset = X_train[:train_size]
        y_train_subset = y_train[:train_size]
        estimator.fit(X_train_subset, y_train_subset)
        y_test_pred = estimator.predict(X_test)
        test_score = accuracy_score(y_test, y_test_pred)
        test_scores_manual.append(test_score)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o--", color="g", label="Validation score")

    # Plot manually computed test scores
    plt.plot(train_sizes, test_scores_manual, "o-", color="b", label="Test score")

    plt.legend(loc="best")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
