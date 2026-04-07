import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def train_model(X, y, lr=0.01, epochs=100):
    """Logistic regression from scratch"""
    theta = np.zeros(X.shape[1])
    m = len(y)

    for _ in range(epochs):
        z = X @ theta
        h = sigmoid(z)
        grad = (X.T @ (h - y)) / m
        theta -= lr * grad

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8

    return theta, mean, std

def predict(X, theta, mean, std):
    """Predict churn probability"""
    X_norm = (X - mean) / std
    return sigmoid(X_norm @ theta)

def auc_score(y_true, y_pred):
    """Simple AUC approximation via trapezoid rule"""
    sorted_idx = np.argsort(y_pred)[::-1]
    y_sorted = y_true[sorted_idx]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg

    auc = np.trapz(tpr, fpr)
    return auc
