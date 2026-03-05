

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], 1))
    for d in range(1, degree + 1):
        X_poly = np.c_[X_poly, X ** d]
    return X_poly


def compute_cost(X, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum((X @ theta - y) ** 2)


def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def main():

    data = pd.read_csv("lab_2/data_02b.csv")
    X = data.iloc[:, 0].values.reshape(-1, 1)
    y = data.iloc[:, 1].values.reshape(-1, 1)

    # Plot feature
    plt.scatter(X, y)
    plt.title("Feature vs Target")
    plt.show()

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    validation_errors = []
    thetas = []

    for d in [1, 2, 3]:
        X_train_poly = add_polynomial_features(X_train, d)
        X_val_poly = add_polynomial_features(X_val, d)

        theta = normal_equation(X_train_poly, y_train)
        error = compute_cost(X_val_poly, y_val, theta)

        validation_errors.append(error)
        thetas.append(theta)

        print(f"Degree {d} Validation Error:", error)

    best_index = np.argmin(validation_errors)
    best_degree = [1, 2, 3][best_index]

    print("\nBEST DEGREE:", best_degree)
    print("Best Parameters:\n", thetas[best_index])

    # Plot fitted curves
    x_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)

    plt.scatter(X, y, label="Data")

    for d, theta in zip([1, 2, 3], thetas):
        X_range_poly = add_polynomial_features(x_range, d)
        y_pred = X_range_poly @ theta
        plt.plot(x_range, y_pred, label=f"d={d}")

    plt.legend()
    plt.title("Polynomial Fits")
    plt.show()

    # Bar plot
    plt.bar([1, 2, 3], validation_errors)
    plt.xlabel("Degree")
    plt.ylabel("Validation Error")
    plt.title("Validation Errors for d=1,2,3")
    plt.show()


if __name__ == "__main__":
    main()