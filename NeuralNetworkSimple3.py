import numpy as np
import pandas as pd

'''
Let's say we have the entire input dataset - X. It has like 10k datapoints (x), each representing
a single 784 pixel picture. Same goes for Y, 10k datapoints (y), each a value 0-9.
For X and Y, it makes sense to have an np.ndarray, since each x and y will have the dimension.

But for the list of weights, where each item is the weight matrix for a particular layer, we need a list.
Same goes for biases, Z, activation values, etc.

Expected Shape(X): 784 * n
Expected Shape(Y): 10 * n

For each hidden layer, the weight matrix is:    size_current_layer * size_prev_layer
The bias matrix is:                             size_current_layer * 1
'''

def activation_function(z: np.ndarray) -> np.ndarray:
    # return np.maximum(z, 0)
    return 1 / (1 + np.exp(-z))


def activation_function_prime(z):
    # return z > 0
    return activation_function(z) * (1 - activation_function(z))


def softmax(z):
    shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def get_predictions(A):
    return np.argmax(A, axis = 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# Here, Y is the one-hot encoded Y, unlike get_accuracy
def calculate_error(A, Y):
    return 0.5 * np.mean(np.sum((A - Y) ** 2, axis = 0))


def initialize_weights(X: np.ndarray, Y: np.ndarray, hidden_layers: list):
    weights = [np.array([0])]   # Weight for the input layer is a 1*1 array with 0
    biases = [np.array([0])]    # Bias for the input layer is a 1*1 array with 0

    # Concatenate(input size, hidden layer sizes, output size) = layer-wise sizes
    layer_sizes = [X.shape[0]] + hidden_layers + [Y.shape[0]]
    L = len(layer_sizes)

    for i in range(1, L):
        weights.append(np.random.rand(layer_sizes[i], layer_sizes[i-1]) - 0.5)
        biases.append(np.random.rand(layer_sizes[i], 1) - 0.5) 

    return weights, biases


def update_parameters(weights: list, biases: list, gradients: list, bias_gradients: list, learning_rate: float):
    l = len(weights)
    for i in range(1, l):
        weights[i] = weights[i] - learning_rate * gradients[i]
        biases[i] = biases[i] - learning_rate * bias_gradients[i]

    return weights, biases


# Returns output (activation from final layer)
# Returns activations per layer
# X takes a batch of input data
def forward_propagate(X: np.ndarray, weights: list, biases: list) -> tuple:
    activations = [X]
    L = len(weights)    # Total number of layers, including input layer
    Z = [0]

    for layer_i in range(1, L):
        z_layer_i = (weights[layer_i] @ activations[layer_i - 1]) + biases[layer_i]
        activations.append(activation_function(z_layer_i))
        Z.append(z_layer_i)

    # Adding a softmax layer doesn't change the results much. Need to understand more.
    activations[L-1] = softmax(activations[L-1])    
    y = activations[L-1]
    return y, activations, Z


def backpropagate(X: np.ndarray, weights: list, biases: list, Y: np.ndarray) -> tuple:
    L = len(weights)
    n = X.shape[1]         # Number of datapoints
    _, activations, Z = forward_propagate(X, weights, biases)

    # Initializing empty gradients list
    gradients = [np.array([0]) for _ in range(L)]
    bias_gradients = [np.array([0]) for _ in range(L)]

    # Calculation for output layer, L
    delta_a_c = activations[L-1] - Y
    output_error = delta_a_c * activation_function_prime(Z[L-1])        # The 1 / n normalization is applied at the next step.
    gradients[L-1] = 1 / n * (output_error @ activations[L-2].T)
    bias_gradients[L-1] = 1 / n * np.sum(output_error, axis = 1, keepdims=True)                   

    delta_l_next = output_error                                         # Error from next layer, delta l+1th layer.
    for l in range(L-2, 0, -1):
        delta_l_current = (weights[l+1].T @ delta_l_next) * activation_function_prime(Z[l])

        # Here, delta_l_current and activations[l-1] is for all datapoints. l * n and (l-1) * n.
        # The dot product sums up all the a * delta values for the datapoints. Therefore, the 1 / n is used to average it out.
        gradients[l] = 1 / n * (delta_l_current @ activations[l-1].T)
        bias_gradients[l] = 1 / n * np.sum(delta_l_current, axis = 1, keepdims=True)

        delta_l_next = delta_l_current

    return gradients, bias_gradients


def train_neural_network(X: np.ndarray, Y: np.ndarray, hidden_layers: list, epochs: int, learning_rate: float) -> tuple:
    weights, biases = initialize_weights(X, Y, hidden_layers)

    for epoch in range(epochs):
        gradients, bias_gradients = backpropagate(X, weights, biases, Y)
        weights, biases = update_parameters(weights, biases, gradients, bias_gradients, learning_rate)

        prev_error = 10000000
        if epoch % 10 == 0:
            a_L, _, _ = forward_propagate(X, weights, biases)
            predictions = get_predictions(a_L)

            print(f"Iteration: {epoch}")
            # Y is one-hot encoded. np.argmax() is used to get the actual Y.
            print(predictions, np.argmax(Y, axis=0))
            print(f'Accuracy: {get_accuracy(predictions, np.argmax(Y, axis=0))}')
            error = calculate_error(a_L, Y)
            print(f'Error   : {error}')

            if (error > prev_error):
                print("Stopping. Reached Oscillation Phase.")
            prev_error = error

    return weights, biases


def main():
    data = pd.read_csv('data/train.csv')
    
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)  # shuffle before splitting into dev and training sets

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.

    transformed_y = [np.zeros([10]) for _ in range(len(Y_train))]
    for i in range(len(Y_train)):
        transformed_y[i][Y_train[i]] = 1

    transformed_y = np.array(transformed_y).T

    train_neural_network(X_train, transformed_y, hidden_layers = [64, 32], epochs = 5000, learning_rate = 50)


if __name__ == '__main__':
    main()