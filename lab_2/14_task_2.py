import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def compute_cost(X, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum((X @ theta - y) ** 2)

def gradient_descent(X_train, y_train, X_val, y_val, alpha=0.01, iterations=300):
    m, n = X_train.shape
    theta = np.zeros((n, 1))
    train_errors, val_errors = [], []

    for _ in range(iterations):
        gradients = (1/m) * X_train.T @ (X_train @ theta - y_train)
        theta -= alpha * gradients
        train_errors.append(compute_cost(X_train, y_train, theta))
        val_errors.append(compute_cost(X_val, y_val, theta))

    return theta, train_errors, val_errors

def k_fold_cv_with_gradients(X, y, k=5, alpha=0.01, iterations=300):
    fold_size = len(X) // k
    val_errors_per_fold = []
    avg_train_curve = np.zeros(iterations)
    avg_val_curve = np.zeros(iterations)
    best_theta = None
    best_val_error = float('inf')

    for i in range(k):
        start, end = i * fold_size, (i+1) * fold_size
        X_val = X[start:end]
        y_val = y[start:end]
        X_train = np.vstack((X[:start], X[end:]))
        y_train = np.vstack((y[:start], y[end:]))

        theta, train_curve, val_curve = gradient_descent(X_train, y_train, X_val, y_val, alpha, iterations)

        val_errors_per_fold.append(val_curve[-1])
        avg_train_curve += np.array(train_curve)
        avg_val_curve += np.array(val_curve)

        if val_curve[-1] < best_val_error:
            best_val_error = val_curve[-1]
            best_theta = theta

        print(f"Fold {i+1} Final Validation Error: {val_curve[-1]:.6f}")

    avg_train_curve /= k
    avg_val_curve /= k

    return best_theta, val_errors_per_fold, avg_train_curve, avg_val_curve


def main():
    data = pd.read_excel("lab_2/Folds5x2_pp.xlsx")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    X_norm_b = add_bias(X_norm)

    best_theta, val_errors_per_fold, avg_train_curve, avg_val_curve = k_fold_cv_with_gradients(
        X_norm_b, y, k=5, alpha=0.01, iterations=300
    )

    print("\nBest Theta from 5-Fold CV:\n", best_theta)

    plt.figure(figsize=(8,5))
    plt.bar(range(1, 6), val_errors_per_fold, color='skyblue')
    plt.xlabel("Fold")
    plt.ylabel("Validation Error")
    plt.title("Validation Error per Fold (5-Fold CV)")
    plt.show()

   
    plt.figure(figsize=(8,5))
    plt.plot(avg_train_curve, label="Average Training Error")
    plt.plot(avg_val_curve, label="Average Validation Error")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Training and Validation Error Curves (Average over 5 Folds)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()