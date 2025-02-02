Using Numpy
"""

import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error Loss Function and its derivative
def mse_loss(pred, target):
    return 0.5 * np.sum((pred - target) ** 2)

def mse_loss_derivative(pred, target):
    return pred - target

# Define the Neural Network structure
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with random values
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights (use small random values for stability)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1  # Small random values
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1

        # Initialize biases
        self.bias_hidden = np.zeros((1, hidden_size))  # Bias for hidden layer
        self.bias_output = np.zeros((1, output_size))  # Bias for output layer

    def forward(self, inputs):
        # Forward pass
        self.inputs = inputs
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)

        return self.output

    def backward(self, inputs, target, learning_rate):
        # Backward pass (gradient descent)
        output_error = mse_loss_derivative(self.output, target) * sigmoid_derivative(self.output)

        hidden_errors = np.dot(output_error, self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_output)

        # Gradient Descent Update for weights and biases
        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_error)
        self.bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)

        self.weights_input_hidden -= learning_rate * np.dot(inputs.T, hidden_errors)
        self.bias_hidden -= learning_rate * np.sum(hidden_errors, axis=0, keepdims=True)

    def train(self, inputs, targets, epochs, learning_rate):
        # Train the network
        for epoch in range(epochs):
            for x, y in zip(inputs, targets):
                x = x.reshape(1, -1)  # Reshape x to 2D array to match input shape (1, input_size)
                y = y.reshape(1, -1)  # Reshape y to 2D array (1, output_size)

                self.forward(x)
                self.backward(x, y, learning_rate)

            if epoch % 100 == 0:
                loss = mse_loss(self.output, y)
                print(f'Epoch {epoch}/{epochs} - Loss: {loss:.4f}')

# Training Data (XOR Problem for example)
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

targets = np.array([0, 1, 1, 0])  # XOR output

# Create and train the neural network
nn = SimpleNN(input_size=2, hidden_size=3, output_size=1)  # Input: 2 neurons, Hidden: 3 neurons, Output: 1 neuron
nn.train(inputs, targets, epochs=10000, learning_rate=0.1)

# Test the network
for x in inputs:
    print(f'Input: {x}, Predicted Output: {nn.forward(x.reshape(1, -1))}')

# Commented out IPython magic to ensure Python compatibility.
# Use %timeit to measure the time taken for training
# %timeit nn.train(inputs, targets, epochs=1000, learning_rate=0.1)



"""Using JAX"""

import jax
import jax.numpy as jnp
from jax import grad, jit

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error Loss Function and its derivative
def mse_loss(pred, target):
    return 0.5 * jnp.sum((pred - target) ** 2)

def mse_loss_derivative(pred, target):
    return pred - target

# Define the Neural Network structure
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with random values
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Using JAX random number generator for better control over initialization
        key = jax.random.PRNGKey(42)
        self.weights_input_hidden = jax.random.normal(key, (input_size, hidden_size)) * 0.1  # Small random values
        self.weights_hidden_output = jax.random.normal(key, (hidden_size, output_size)) * 0.1

        self.bias_hidden = jnp.zeros(hidden_size)
        self.bias_output = jnp.zeros(output_size)

    def forward(self, inputs):
        # Forward pass
        self.inputs = inputs
        self.hidden_input = jnp.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.output_input = jnp.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)

        return self.output

    def backward(self, inputs, target, learning_rate):
        # Backward pass (gradient descent)
        output_error = mse_loss_derivative(self.output, target) * sigmoid_derivative(self.output)

        hidden_errors = jnp.dot(output_error, self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_output)

        # Gradient Descent Update for weights and biases
        self.weights_hidden_output -= learning_rate * jnp.outer(self.hidden_output, output_error)
        self.bias_output -= learning_rate * output_error

        self.weights_input_hidden -= learning_rate * jnp.outer(inputs, hidden_errors)
        self.bias_hidden -= learning_rate * hidden_errors

    def train(self, inputs, targets, epochs, learning_rate):
        # Train the network
        for epoch in range(epochs):
            for x, y in zip(inputs, targets):
                self.forward(x)
                self.backward(x, y, learning_rate)

            if epoch % 100 == 0:
                loss = mse_loss(self.output, y)
                print(f'Epoch {epoch}/{epochs} - Loss: {loss:.4f}')

# Training Data (XOR Problem for example)
inputs = jnp.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

targets = jnp.array([0, 1, 1, 0])  # XOR output

# Create and train the neural network
nn = SimpleNN(input_size=2, hidden_size=3, output_size=1)  # Input: 2 neurons, Hidden: 3 neurons, Output: 1 neuron
nn.train(inputs, targets, epochs=10000, learning_rate=0.1)

# Test the network
for x in inputs:
    print(f'Input: {x}, Predicted Output: {nn.forward(x)}')

# Commented out IPython magic to ensure Python compatibility.
# Use %timeit to measure the time taken for training
# %timeit nn.train(inputs, targets, epochs=1000, learning_rate=0.1)


