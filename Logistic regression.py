import numpy as np
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = X.shape[0]
    for _ in range(iterations):
        predictions = sigmoid(np.dot(X, weights))
        errors = predictions - y
        gradient = np.dot(X.T, errors) / m
        weights -= learning_rate * gradient
    return weights
def predict(X, weights):
    return sigmoid(np.dot(X, weights)) >= 0.5

# Logistic regression training
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    weights = np.zeros(X.shape[1])
    weights = gradient_descent(X, y, weights, learning_rate, iterations)
    return weights
def evaluate(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred) * 100
    return accuracy
X = np.array([[0.1, 1.1], [1.2, 0.9], [1.5, 1.6], [2.0, 1.8], [2.5, 2.1],
              [0.5, 1.5], [1.8, 2.3], [0.2, 0.7], [1.9, 1.4], [0.8, 0.6]])
y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std
weights = logistic_regression(X_normalized, y, learning_rate=0.1, iterations=1000)
X_with_bias = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))
y_pred = predict(X_with_bias, weights)
accuracy = evaluate(y, y_pred)
loss = cross_entropy_loss(y, sigmoid(np.dot(X_with_bias, weights)))
print(f"Accuracy: {accuracy:.2f}%")
print(f"Cross-Entropy Loss: {loss:.4f}")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
XX = np.c_[np.ones((xx1.ravel().shape[0], 1)), (np.c_[xx1.ravel(), xx2.ravel()] - X_mean) / X_std]
Z = predict(XX, weights).reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.2, cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()
