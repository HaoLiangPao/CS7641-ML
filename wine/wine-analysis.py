import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# 1. Load the wine quality dataset
red_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    delimiter=";",
)
white_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    delimiter=";",
)

# Combine red and white wine datasets and add a column to distinguish them
red_wine["type"] = 0
white_wine["type"] = 1
wine_data = pd.concat([red_wine, white_wine])

# Check the first few rows of the dataset
print(wine_data.head())


# 2. Process Wine Dataset
from sklearn.model_selection import train_test_split

# Split into features and labels
X = wine_data.drop("quality", axis=1)
y = wine_data["quality"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 3. Build and train models

# NN
def neural_networks_analysis():
    # For Wine Quality
    model_wine = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model_wine.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model_wine.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # For Fashion MNIST
    model_fashion = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model_fashion.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model_fashion.fit(ds_train, epochs=10, validation_data=ds_test)

# SVM
def support_vector_machine_analysis():
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


# KNeighbors
def kNeighbors_analysis():
    # For Wine Quality
    knn_wine = KNeighborsClassifier(n_neighbors=5)
    knn_wine.fit(X_train, y_train)
    print(knn_wine.score(X_test, y_test))

    # For Fashion MNIST
    knn_fashion = KNeighborsClassifier(n_neighbors=5)
    knn_fashion.fit(X_train_fashion, y_train_fashion)
    print(knn_fashion.score(X_test_fashion, y_test_fashion))
