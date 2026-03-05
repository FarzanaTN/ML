

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std


def compute_cost(X, y, theta):
    m = len(y)
    return (1/(2*m)) * np.sum((X @ theta - y)**2)


def gradient_descent_with_validation(X_train, y_train, X_val, y_val,
                                     alpha=0.01, iterations=300):

    m, n = X_train.shape
    theta = np.zeros((n, 1))

    train_errors = []
    val_errors = []

    for _ in range(iterations):

        gradients = (1/m) * X_train.T @ (X_train @ theta - y_train)
        theta -= alpha * gradients

        train_cost = compute_cost(X_train, y_train, theta)
        val_cost = compute_cost(X_val, y_val, theta)

        train_errors.append(train_cost)
        val_errors.append(val_cost)

    return theta, train_errors, val_errors



def main():

    data = pd.read_excel("lab_2/Folds5x2_pp.xlsx")

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

   
    for i in range(X.shape[1]):
        plt.figure()
        plt.scatter(X[:, i], y)
        plt.xlabel(f"Feature {i+1}")
        plt.ylabel("Target")
        plt.title(f"Feature {i+1} vs Target")
        plt.show()

   
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # ======================================================
    # WITHOUT NORMALIZATION
    # ======================================================

    X_train_b = add_bias(X_train)
    X_val_b = add_bias(X_val)

    theta, train_curve, val_curve = gradient_descent_with_validation(
        X_train_b, y_train, X_val_b, y_val,
        alpha=0.00000001,  
        iterations=300
    )

    print("\nWITHOUT NORMALIZATION")
    print("Final Training Error:", train_curve[-1])
    print("Final Validation Error:", val_curve[-1])
    print("Parameters:\n", theta)

    plt.figure()
    plt.plot(train_curve, label="Training Error")
    plt.plot(val_curve, label="Validation Error")
    plt.legend()
    plt.title("Without Normalization")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

   

    X_train_norm, mean, std = normalize(X_train)
    X_val_norm = (X_val - mean) / std

    X_train_norm_b = add_bias(X_train_norm)
    X_val_norm_b = add_bias(X_val_norm)

    theta_norm, train_curve_norm, val_curve_norm = gradient_descent_with_validation(
        X_train_norm_b, y_train, X_val_norm_b, y_val,
        alpha=0.01,
        iterations=300
    )

    print("\nWITH NORMALIZATION")
    print("Final Training Error:", train_curve_norm[-1])
    print("Final Validation Error:", val_curve_norm[-1])
    print("Parameters:\n", theta_norm)

    # plt.figure()
    # plt.plot(train_curve_norm, label="Training Error")
    # plt.plot(val_curve_norm, label="Validation Error")
    # plt.legend()
    # plt.title("With Normalization")
    # plt.xlabel("Iterations")
    # plt.ylabel("Cost")
    # plt.show()


if __name__ == "__main__":
    main()