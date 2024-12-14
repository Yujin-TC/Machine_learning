import numpy as np
from scipy.misc import derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

x = np.array([3.8481, 10.1539, -3.8561, -4.2228])  # Input features
y = 0  # True output

np.random.seed(42)  # Seed for reproducibility
W1 = np.random.randn(4, 3)  # Weights from input layer to hidden layer (4x3 matrix)
b1 = np.random.randn(1, 3)  # Biases for hidden layer (1x3 vector)

W2 = np.random.randn(3, 1)  # Weights from hidden layer to output layer (3x1 vector)
b2 = np.random.randn(1, 1)  # Bias for output layer (1x1 scalar)

# Learning rate
lr = 0.01
z_h = np.dot(x, W1) + b1
a_h = sigmoid(z_h)  

z_o = np.dot(a_h, W2) + b2
a_o = sigmoid(z_o)  # Apply sigmoid activation function for output

error = 0.5 * (a_o - y) ** 2
derivative_error_a_o = a_o - y  # Derivative of error with respect to a_o
derivative_a_o_z_o = sigmoid_derivative(a_o)  # Derivative of sigmoid function
derivative_error_z_o = derivative_error_a_o * derivative_a_o_z_o  # Gradient of error w.r.t. z_o

derivative_error_W2 = np.dot(a_h.reshape(-1, 1), derivative_error_z_o.reshape(1, -1))
derivative_error_b2 = derivative_error_z_o

derivative_a_h_z_h = sigmoid_derivative(a_h)  # Derivative of sigmoid function for hidden layer
derivative_error_z_h = np.dot(derivative_error_z_o, W2.T) * derivative_a_h_z_h  # Gradient of error w.r.t. z_h

# Compute gradients for W1 and b1
derivative_error_W1 = np.dot(x.reshape(-1, 1), derivative_error_z_h.reshape(1, -1))
derivative_error_b1 = derivative_error_z_h

# Update weights and biases using gradient descent
W1 -= lr * derivative_error_W1
b1 -= lr * derivative_error_b1
W2 -= lr * derivative_error_W2
b2 -= lr * derivative_error_b2

# Print results
print("Updated weights and biases:")
print("Weights from input layer to first hidden layer :", W1)
print("b1:", b1)
print("Weights from hidden layer to output layer:", W2)
print("b2:", b2)
