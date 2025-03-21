import numpy as np
import matplotlib.pyplot as plt

def init_params(layer_dims):
    """
    Initialize parameters for the neural network.
    layer_dims: list containing the number of units in each layer.
    Returns a dictionary with weights 'W1', 'W2', ..., and biases 'b1', 'b2', ...
    """
    params = {}
    np.random.seed(3)  # ensures the same random values are generated every time
    L = len(layer_dims)
    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return params 

def sigmoid(Z):
    """
    Computes the sigmoid activation.
    Returns:
      A: the sigmoid of Z.
      cache: the input Z, stored for backward propagation.
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def forward_prop(X, params):
    """
    Implements forward propagation for the network.
    X: input data.
    params: dictionary of network parameters.
    Returns the final activation A and a list of caches for each layer.
    """
    A = X  # initial activation is the input data
    caches = []
    L = len(params) // 2  # number of layers
    for l in range(1, L + 1):
        A_prev = A
        Z = np.dot(params['W' + str(l)], A_prev) + params['b' + str(l)]
        linear_cache = (A_prev, params['W' + str(l)], params['b' + str(l)])
        A, activation_cache = sigmoid(Z)
        cache = (linear_cache, activation_cache)
        caches.append(cache)
    return A, caches

def cost_function(A, Y):
    """
    Computes the cross-entropy cost.
    A: predicted output from the network.
    Y: true labels.
    Returns the cost.
    """
    m = Y.shape[1]
    cost = (-1 / m) * (np.dot(np.log(A), Y.T) + np.dot(np.log(1 - A), (1 - Y).T))
    return np.squeeze(cost)

def one_layer_backward(dA, cache):
    """
    Implements the backward propagation for one layer.
    dA: gradient of the cost with respect to the activation of this layer.
    cache: tuple of (linear_cache, activation_cache) from forward propagation.
    Returns the gradients dA_prev, dW, db.
    """
    linear_cache, activation_cache = cache
    Z = activation_cache
    # Compute the sigmoid activation once and extract its value.
    s, _ = sigmoid(Z)
    # Derivative of the sigmoid function.
    dZ = dA * s * (1 - s)
    
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backprop(AL, Y, caches):
    """
    Implements the backward propagation for the entire network.
    AL: output of forward propagation.
    Y: true labels.
    caches: list of caches from forward propagation.
    Returns a dictionary with gradients for each parameter.
    """
    grads = {}
    L = len(caches)         # total number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # ensure Y has the same shape as AL
    
    # Compute gradient of the cost with respect to AL (using cross-entropy derivative).
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Backprop for the output layer (layer L)
    current_cache = caches[L - 1]
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = one_layer_backward(dAL, current_cache)
    
    # Loop over the hidden layers in reverse order.
    for l in reversed(range(1, L)):
        current_cache = caches[l - 1]
        dA_curr = grads["dA" + str(l + 1)]
        dA_prev_temp, dW_temp, db_temp = one_layer_backward(dA_curr, current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l)] = dW_temp
        grads["db" + str(l)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent.
    parameters: dictionary containing current weights and biases.
    grads: dictionary containing gradients for each parameter.
    learning_rate: the learning rate for gradient descent.
    Returns updated parameters.
    """
    L = len(parameters) // 2  # number of layers

    for l in range(1, L + 1):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]
    return parameters

def train(X, Y, layer_dims, epochs, lr):
    """
    Trains the neural network.
    X: input data.
    Y: true labels.
    layer_dims: list containing the number of units in each layer.
    epochs: number of iterations to train.
    lr: learning rate.
    Returns the trained parameters and the history of cost values.
    """
    params = init_params(layer_dims)
    cost_history = []

    for i in range(epochs):
        Y_hat, caches = forward_prop(X, params)
        cost = cost_function(Y_hat, Y)
        cost_history.append(np.squeeze(cost))
        grads = backprop(Y_hat, Y, caches)
        params = update_parameters(params, grads, lr)

    return params, cost_history

# ------------------- Running the Training -------------------

# Define a simple neural network architecture:
# - 2 input features
# - 1 hidden layer with 4 neurons
# - 1 output neuron (binary classification)
layer_dims = [2, 4, 1]

# Generate some dummy training data:
m = 500  # number of examples
X = np.random.randn(layer_dims[0], m)  # shape: (2, 500)
Y = (np.random.rand(1, m) > 0.5).astype(int)  # shape: (1, 500)

# Training hyperparameters:
epochs = 1000         # number of iterations
learning_rate = 0.01  # learning rate

# Train the model:
params, cost_history = train(X, Y, layer_dims, epochs, learning_rate)

# Print the final parameters and the last few cost values to monitor convergence.
print("Trained Parameters:")
for key, value in params.items():
    print(f"{key}: {value.shape}")

print("\nCost History (last 5 values):")
print(cost_history[-5:])


# Optional: Plot the cost history to visualize convergence.
plt.plot(cost_history)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost History")
plt.show()
