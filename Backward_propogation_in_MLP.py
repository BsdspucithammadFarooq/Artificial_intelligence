import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.1, 0.6], [0.15, 0.71], [0.25, 0.8], [0.35, 0.45],
              [0.5, 0.5], [0.6, 0.2], [0.65, 0.3], [0.8, 0.35]])
y = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])

#Initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(1)
    weights = {
        'W1': np.random.randn(input_size, hidden_size),
        'b1': np.zeros((1, hidden_size)),
        'W2': np.random.randn(hidden_size, output_size),
        'b2': np.zeros((1, output_size))
    }
    return weights
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def forward_propagation(X, weights):
    Z1 = np.dot(X, weights['W1']) + weights['b1']
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, weights['W2']) + weights['b2']
    A2 = sigmoid(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
def backward_propagation(X, y, weights, cache):
    Z1, A1, Z2, A2 = cache
    m = X.shape[0]
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, weights['W2'].T)
    dZ1 = dA1 * (A1 * (1 - A1))
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return gradients

#Update weights
def update_parameters(weights, gradients, learning_rate):
    weights['W1'] -= learning_rate * gradients['dW1']
    weights['b1'] -= learning_rate * gradients['db1']
    weights['W2'] -= learning_rate * gradients['dW2']
    weights['b2'] -= learning_rate * gradients['db2']
    return weights

# Step 7: Training loop
def train_network(X, y, hidden_size, learning_rate, epochs):
    input_size = X.shape[1]
    output_size = 1
    weights = initialize_parameters(input_size, hidden_size, output_size)

    for i in range(epochs):
        y_pred, cache = forward_propagation(X, weights)
        loss = compute_loss(y, y_pred)
        gradients = backward_propagation(X, y, weights, cache)
        weights = update_parameters(weights, gradients, learning_rate)

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.4f}")
    return weights
def plot_decision_boundary(X, y, weights):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = forward_propagation(np.c_[xx.ravel(), yy.ravel()], weights)[0]
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolor='k')
    plt.title('Decision Boundary')
    plt.show()
weights = train_network(X, y, hidden_size=4, learning_rate=1, epochs=1000)
plot_decision_boundary(X, y, weights)
