# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# =========================
# GIVEN FUNCTIONS (DO NOT CHANGE)
# =========================
def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def misclassification_error(y_true, y_pred_probs, threshold=0.5):
    y_pred_labels = (y_pred_probs >= threshold).astype(int)
    return np.mean(y_pred_labels != y_true)

# =========================
# Data Loader
# =========================
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    return pd.read_csv(url, header=None, names=cols)

# =========================
# Perceptron
# =========================
class Perceptron:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X_train, y_train, X_val, y_val):
        self.w = np.zeros(X_train.shape[1])
        self.b = 0

        self.train_errors = []
        self.val_errors = []

        y_train_mod = np.where(y_train == 0, -1, 1)

        for _ in range(self.epochs):
            # training update
            for i in range(len(X_train)):
                linear = np.dot(X_train[i], self.w) + self.b
                pred = 1 if linear >= 0 else -1

                if pred != y_train_mod[i]:
                    self.w += self.lr * y_train_mod[i] * X_train[i]
                    self.b += self.lr * y_train_mod[i]

            # training error
            train_pred = (np.dot(X_train, self.w) + self.b >= 0).astype(int)
            train_err = np.mean(train_pred != y_train)

            # validation error
            val_pred = (np.dot(X_val, self.w) + self.b >= 0).astype(int)
            val_err = np.mean(val_pred != y_val)

            self.train_errors.append(train_err)
            self.val_errors.append(val_err)

    def predict(self, X):
        return (np.dot(X, self.w) + self.b >= 0).astype(int)

# =========================
# Logistic Regression
# =========================
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X_train, y_train, X_val, y_val):
        n, d = X_train.shape
        self.w = np.zeros(d)
        self.b = 0

        self.train_losses = []
        self.val_losses = []

        self.train_errors = []
        self.val_errors = []

        for _ in range(self.epochs):   # ONLY LOOP
            # forward
            z = X_train @ self.w + self.b
            y_pred = self.sigmoid(z)

            # gradients (vectorized)
            self.w -= self.lr * (1/n) * (X_train.T @ (y_pred - y_train))
            self.b -= self.lr * (1/n) * np.sum(y_pred - y_train)

            # training metrics
            self.train_losses.append(log_loss(y_train, y_pred))
            self.train_errors.append(misclassification_error(y_train, y_pred))

            # validation metrics
            val_probs = self.sigmoid(X_val @ self.w + self.b)
            self.val_losses.append(log_loss(y_val, val_probs))
            self.val_errors.append(misclassification_error(y_val, val_probs))

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# =========================
# Evaluation
# =========================
def evaluate(y_true, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

# =========================
# MAIN
# =========================
data = load_data()

X = data.drop("Outcome", axis=1).values
y = data["Outcome"].values

# scale features
X = StandardScaler().fit_transform(X)

# split (train + validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# Perceptron
# =========================
p = Perceptron(lr=0.01, epochs=100)
p.fit(X_train, y_train, X_val, y_val)

plt.plot(p.train_errors, label="Train")
plt.plot(p.val_errors, label="Validation")
plt.title("Perceptron: Misclassification vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.legend()
plt.show()

y_pred = p.predict(X_val)
print("\nPerceptron Results")
evaluate(y_val, y_pred)

# =========================
# Logistic Regression
# =========================
lr = LogisticRegressionScratch(lr=0.01, epochs=100)
lr.fit(X_train, y_train, X_val, y_val)

# Misclassification curve
plt.plot(lr.train_errors, label="Train")
plt.plot(lr.val_errors, label="Validation")
plt.title("Logistic Regression: Misclassification")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.legend()
plt.show()

# Log Loss curve
plt.plot(lr.train_losses, label="Train")
plt.plot(lr.val_losses, label="Validation")
plt.title("Logistic Regression: Log Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

y_pred = lr.predict(X_val)
print("\nLogistic Regression Results")
evaluate(y_val, y_pred)

# =========================
# Naive Bayes
# =========================
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_val)
print("\nNaive Bayes Results")
evaluate(y_val, y_pred)