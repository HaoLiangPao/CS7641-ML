from sklearn.base import BaseEstimator, ClassifierMixin


class CustomKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn=None, epochs=1, batch_size=32, verbose=0, **kwargs):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y, **kwargs):
        if self.build_fn is None:
            raise ValueError("A build function must be provided to build the model")
        self.model = self.build_fn(**self.kwargs)
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            **kwargs
        )
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("The model must be built and trained before prediction")
        return (self.model.predict(X) > 0.5).astype("int32")

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("The model must be built and trained before prediction")
        return self.model.predict(X)

    def score(self, X, y):
        if self.model is None:
            raise ValueError("The model must be built and trained before scoring")
        return self.model.evaluate(X, y, verbose=self.verbose)[1]
