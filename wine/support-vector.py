from sklearn.svm import SVC


# For Wine Quality
svm_wine = SVC(kernel="linear")
svm_wine.fit(X_train, y_train)
print(svm_wine.score(X_test, y_test))

# For Fashion MNIST (flatten the images)
import numpy as np

X_train_fashion = np.concatenate([x.numpy().flatten() for x, _ in ds_train])
y_train_fashion = np.concatenate([y.numpy() for _, y in ds_train])
X_test_fashion = np.concatenate([x.numpy().flatten() for x, _ in ds_test])
y_test_fashion = np.concatenate([y.numpy() for _, y in ds_test])

svm_fashion = SVC(kernel="linear")
svm_fashion.fit(X_train_fashion, y_train_fashion)
print(svm_fashion.score(X_test_fashion, y_test_fashion))
