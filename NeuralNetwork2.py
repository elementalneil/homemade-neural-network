import random
import numpy as np
import pickle
import gzip


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1 - sigmoid(z))


def normalize(z: np.ndarray) -> np.ndarray:
    # return z / np.sum(z)
    return np.exp(z) / sum(np.exp(z))

def calculate_error(y: np.ndarray, expected_y: np.ndarray) -> float:
    magnitude = np.linalg.norm(y - expected_y) ** 2
    return magnitude / (2 * len(y))


# Returns output (activation from final layer)
# Returns activations per layer
def forward_propagate(x: np.ndarray, weights: list, biases: list) -> tuple:
    activations = [x]
    L = len(weights)    # Total number of layers, including input layer
    Z = [0]

    for layer_i in range(1, L):
        z_layer_i = (weights[layer_i] @ activations[layer_i - 1]) + biases[layer_i]
        activations.append(sigmoid(z_layer_i))
        Z.append(z_layer_i)

    y = activations[L-1]
    return y, activations, Z


# weights should be a list of 2D numpy arrays
# biases should be a list of 1D numpy arrays
# IMPORTANT: Assuming that both weights and biases are there for every layer, including input layer
# Since weight for input layer makes zero sense, it can be zero, None or whatever
# returns weight and bias gradients
# WARNING: IS THIS NECESSARY? expects y to be normalized (probabilities)
def backpropagate(x: np.ndarray, weights: list, biases: list, y: np.ndarray) -> tuple:
    # activations = [x]
    # L = len(weights) + 1    # Total number of layers, including input layer
    #
    # for layer_i in range(1, L):
    #     z_layer_i = weights[layer_i] @ activations[layer_i - 1] + biases[layer_i]
    #     activations.append(sigmoid(z_layer_i))

    L = len(weights)     # Total number of layers, including input layer
    _, activations, Z = forward_propagate(x, weights, biases)

    # WARNING: IS NORMALIZATION NECESSARY ???
    normalized_output = normalize(activations[L-1])
    delta_a_c = normalized_output - y
    # delta_a_c = activations[L-1] - y

    gradients = [np.array([0]) for _ in range(L)]
    bias_gradients = [np.array([0]) for _ in range(L)]

    # Note: In numpy, use * for Hadamard product, and @ for matrix multiplication
    # activation, a = sigmoid(z), so, sigmoid_prime(z) = (sigmoid(z) * (1-sigmoid(z)) = a * (1-a)
    # output_error = delta_a_c * (activations[L-1] * (1-activations[L-1]))
    output_error = delta_a_c * sigmoid_prime(Z[L-1])

    activations_transpose = activations[L-2].reshape(-1, 1)  # Converts [1, 2, 3] to [[1],[2],[3]]
    transformed_output_error = np.array([output_error])      # Converts [1, 2, 3] to [[1, 2, 3]]
    gradients[L-1] = (activations_transpose @ transformed_output_error).T
    # gradients[L-1] = output_error.reshape(-1, 1) @ np.array([activations[L-2]])
    bias_gradients[L-1] = output_error

    for l in range(L-2, 0, -1):
        # output_error = (weights[l+1].T @ output_error) * (activations[l] * (1-activations[l]))
        output_error = (weights[l+1].T @ output_error) * (sigmoid_prime(Z[l]))

        activations_transpose = activations[l-1].reshape(-1, 1)
        transformed_output_error = np.array([output_error])
        gradients[l] = (activations_transpose @ transformed_output_error).T
        # gradients[l] = output_error.reshape(-1, 1) @ np.array([activations[l-1]])
        bias_gradients[l] = output_error

    return gradients, bias_gradients


# Returns weights and biases
def train_neural_network(x: list, y: list, hidden_layers: list, epochs: int, learning_rate: float) -> tuple:
    weights = [np.array([0])]   # Weight for the input layer is a 1*1 array with 0
    biases = [np.array([0])]    # Bias for the input layer is a 1*1 array with 0
    layer_sizes = [len(x[0])] + hidden_layers + [len(y[0])]
    L = len(layer_sizes)

    for i in range(1, L):
        weights.append(np.random.randn(layer_sizes[i], layer_sizes[i-1]))
        biases.append(np.random.randn(layer_sizes[i]))

    for epoch in range(epochs):
        weight_gradient_sum = [np.array([0])]
        bias_gradient_sum = [np.array([0])]

        for l in range(1, L):
            weight_gradient_sum.append(np.zeros([layer_sizes[l], layer_sizes[l-1]]))
            bias_gradient_sum.append(np.zeros(layer_sizes[l]))

        for x_, y_ in zip(x, y):
            weight_gradient, bias_gradient = backpropagate(x_, weights, biases, y_)
            for i in range(1, L):
                weight_gradient_sum[i] += weight_gradient[i]
                bias_gradient_sum[i] += bias_gradient[i]

        weight_gradient_avg = [arr / len(x) for arr in weight_gradient_sum]
        bias_gradient_avg = [arr / len(x) for arr in bias_gradient_sum]

        for i in range(1, L):
            weights[i] -= learning_rate * weight_gradient_avg[i]
            biases[i] -= learning_rate * bias_gradient_avg[i]

        total_error = 0.0
        for x_, y_ in zip(x, y):
            observed_y, _, _ = forward_propagate(x_, weights, biases)
            total_error += calculate_error(normalize(observed_y), y_)
        error = total_error / len(x)
        print(f'Error: {error} after epoch {epoch + 1}')

    return weights, biases


def validate_neural_network(weights: list, biases: list, x_test: list, y_test: list) -> tuple:
    correct_count = 0
    for x_, y_ in zip(x_test, y_test):
        observed_y, _, _ = forward_propagate(x_, weights, biases)
        observed_y = normalize(observed_y)

        observed_value = np.argmax(observed_y)
        actual_value = np.argmax(y_)
        if observed_y[observed_value] >= 0.1:
            print(f'Predicted: {observed_value} | Actual: {actual_value}')
            if observed_value == actual_value: correct_count += 1
        else:
            print('Unable to predict')

    print(f'Accuracy: {correct_count / len(x_test)}')


def dummy_data_generator(weight1, weight2, bias, count = 1000, max_value = 5):
    X = []
    Y = []
    # Explicit value for deterministic execution
    X.append(np.array([1, 1]))
    Y.append(np.array([sigmoid(sigmoid(weight1 * 1 + weight2 * 1 + bias))]))
    for i in range(count):
        x1 = random.randint(0, max_value)
        x2 = random.randint(0, max_value)
        y = sigmoid(sigmoid(weight1 * x1 + weight2 * x2 + bias))
        X.append(np.array([x1, x2]))
        Y.append(np.array([y]))
    return X, Y


def main():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="bytes")
    f.close()

    X = training_data[0]
    Y = training_data[1]
    transformed_y = [np.zeros([10]) for _ in range(len(Y))]
    for i in range(len(Y)):
        transformed_y[i][Y[i]] = 1

    hidden_layers_arr = [30]

    weights, biases = train_neural_network(X, transformed_y, hidden_layers_arr, 30, 3.0)
    validate_neural_network(weights, biases, X[:50], transformed_y[:50])


def trivial_runner():
    X, Y = dummy_data_generator(-1, 2, -1)
    weights, biases = train_neural_network(X, Y, [1], 20000, 1.0)
    for i in range(50):
        y, _ = forward_propagate(X[i], weights, biases)
        print(f'Predicted: {y} | Actual: {Y[i]}')


if __name__ == '__main__':
    main()