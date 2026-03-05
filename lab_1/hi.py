import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

def load_data(file_path="lab_1/data_01.csv", synthetic=False):
    
    if synthetic:
        x = np.arange(1, 101).reshape(-1, 1)  # x = 1 to 100
        noise = np.random.randn(100, 1)       # Gaussian noise N(0,1)
        y = 3 + 5 * x + noise
        data = np.hstack((x, y))
        df = pd.DataFrame(data, columns=["x", "y"])
        df.to_csv("lab_1/lab01_data.csv", index=False)
        print("Synthetic data saved to lab01_data.csv")
    else:
        df = pd.read_csv(file_path)    
    return df



def process_data(df, scale=False):
    
    X = df["x"].values.reshape(-1, 1)
    y = df["y"].values.reshape(-1, 1)
    
    if scale:
        # X = (X - np.mean(X)) / np.std(X)
        X = (X - np.mean(X)) / np.std(X)
    
    # Add dummy feature x0 = 1
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    return X, y


# how wrong the model is? J(thjeta)
def compute_cost(X, y, theta):
    
    m = len(y)
    predictions = X @ theta
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost

# using alpha (learning rate)
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradients = (1/m) * (X.T @ errors)
        theta = theta - alpha * gradients
        
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history











def train(X, y, alpha=0.0001, iterations=1000):
    theta = np.zeros((X.shape[1], 1))
    theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
    return theta, cost_history



def evaluate(X, y, theta):
    predictions = X @ theta
    mse = np.mean((predictions - y) ** 2)
    return mse



if __name__ == "__main__":

    # -------- Real Data --------
    df = load_data(synthetic=False)

    # Process data
    X, y = process_data(df, scale=False)

    # Plot real data
    plt.scatter(df["x"], df["y"])
    plt.title("Real Data Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Train model
    theta, cost_history = train(X, y, alpha=0.0001, iterations=1000000)

    print("\nLearned Parameters (theta):")
    print(f"theta0 (bias) = {theta[0][0]}")
    print(f"theta1 (slope) = {theta[1][0]}")

    # Plot training error curve
    plt.plot(cost_history)
    plt.title("Training Error vs Iterations (Real Data)")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

    # Plot regression line
    x_vals = df["x"].values
    X_plot = np.hstack((np.ones((len(x_vals), 1)), x_vals.reshape(-1, 1)))
    y_pred = X_plot @ theta

    plt.scatter(df["x"], df["y"], label="Real Data")
    plt.plot(df["x"], y_pred, color="red", label="Learned Regression Line")
    plt.legend()
    plt.title("Linear Regression on Real Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Evaluate model
    mse = evaluate(X, y, theta)
    print(f"\nFinal MSE (Real Data): {mse}") 