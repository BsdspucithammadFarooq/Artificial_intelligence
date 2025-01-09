import numpy as np
import matplotlib.pyplot as plt

# Perceptron Implementation
def perceptron(X, y, learning_rate, epochs):
    weights = np.zeros(X.shape[1])
    bias = 0
    for epoch in range(epochs):
        for i in range(X.shape[0]):
            prediction = np.dot(X[i], weights) + bias
            if y[i] * prediction <= 0:
                weights += learning_rate * y[i] * X[i]
                bias += learning_rate * y[i]
    return weights, bias
def plot_perceptron_boundary(X, y, weights, bias):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title('Perceptron Decision Boundary')
    plt.show()
# XOR Neural Network Implementation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def train_xor_nn(X, y, hidden_size=4, learning_rate=1, epochs=10000):
    np.random.seed(1)
    input_size = X.shape[1]
    output_size = 1

    weights = {
        'W1': np.random.randn(input_size, hidden_size),
        'b1': np.zeros((1, hidden_size)),
        'W2': np.random.randn(hidden_size, output_size),
        'b2': np.zeros((1, output_size))
    }
    for i in range(epochs):
        #fow prog
        Z1 = np.dot(X, weights['W1']) + weights['b1']
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, weights['W2']) + weights['b2']
        A2 = sigmoid(Z2)
        # Compute loss
        loss = -np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2))
        # Back pro
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2) / X.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
        dA1 = np.dot(dZ2, weights['W2'].T)
        dZ1 = dA1 * (A1 * (1 - A1))
        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
        # Update weights
        weights['W1'] -= learning_rate * dW1
        weights['b1'] -= learning_rate * db1
        weights['W2'] -= learning_rate * dW2
        weights['b2'] -= learning_rate * db2
    return weights
def visualize_xor_boundary(weights, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z1 = np.dot(np.c_[xx.ravel(), yy.ravel()], weights['W1']) + weights['b1']
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, weights['W2']) + weights['b2']
    A2 = sigmoid(Z2)
    Z = np.round(A2).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolor='k')
    plt.title('XOR Neural Network Decision Boundary')
    plt.show()
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])
X_perceptron = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_perceptron = np.array([0, 0, 0, 1]) * 2 - 1
# Train Perceptron and Plot
weights, bias = perceptron(X_perceptron, y_perceptron, learning_rate=0.1, epochs=100)
plot_perceptron_boundary(X_perceptron, y_perceptron, weights, bias)
# Train XOR Neural Network and Plot
trained_weights = train_xor_nn(X_xor, y_xor)
visualize_xor_boundary(trained_weights, X_xor, y_xor)
