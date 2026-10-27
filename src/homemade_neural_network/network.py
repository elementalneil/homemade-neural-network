from pathlib import Path

import numpy as np

from enum import Enum
from collections.abc import Callable
from dataclasses import dataclass


# Dataclass is equivalent to a C# Record; Immutable + Automatic Constructor Creation
@dataclass(frozen=True)
class TrainingProgress:
    completed_epochs: int
    total_epochs: int
    accuracy: float | None = None
    error: float | None = None


class ActivationFunctionValues(Enum):
    SIGMOID = "sigmoid"
    RELU = "ReLU"


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

-------------------
ACTIVATION FUNCTION
-------------------
For ReLU, have a smaller learning rate (1/10th) compared to Sigmoid.
Typically, for ReLU, use 0.1 - 1.0, and for Sigmoid, use 1.0 - 10.
'''


class NeuralNetwork:
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        hidden_layers: list,
        epochs: int = 1000,
        learning_rate: float = 0.1,
        activation_function_type: ActivationFunctionValues = ActivationFunctionValues.SIGMOID
    ):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.hidden_layers = list(hidden_layers)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.parameters = None

        if not isinstance(activation_function_type, ActivationFunctionValues):
            print("Invalid value for activation_function_type. Defaulting to Sigmoid.")
            self._activation_function_type = ActivationFunctionValues.SIGMOID
        else:
            self._activation_function_type = activation_function_type


    # region Pure Methods (TODO: Consider marking static)

    def _activation_function(self, z: np.ndarray) -> np.ndarray:
        if self._activation_function_type == ActivationFunctionValues.SIGMOID:
            return 1 / (1 + np.exp(-z))
        else:
            return np.maximum(z, 0)


    def _activation_function_prime(self, z: np.ndarray) -> np.ndarray:
        if self._activation_function_type == ActivationFunctionValues.SIGMOID:
            return self._activation_function(z) * (1 - self._activation_function(z))
        else:
            return (z > 0).astype(float)        


    def _softmax(self, z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)


    def get_predictions(self, A: np.ndarray) -> np.ndarray:
        return np.argmax(A, axis = 0)


    # Expects A and Y to be the predicted class labels, instead of class probabilities.
    def get_accuracy(self, A: np.ndarray, Y_: np.ndarray) -> float:
        return np.sum(A == Y_) / Y_.size


    # Here, Y is the one-hot encoded Y, unlike get_accuracy
    def _calculate_error(self, A, Y_):
        return 0.5 * np.mean(np.sum((A - Y_) ** 2, axis = 0))

    # endregion


    # region Private Methods

    def _initialize_weights(self) -> tuple:
        weights = [np.array([0])]   # Weight for the input layer is a 1*1 array with 0
        biases = [np.array([0])]    # Bias for the input layer is a 1*1 array with 0

        # Concatenate(input size, hidden layer sizes, output size) = layer-wise sizes
        layer_sizes = [self.X.shape[0], *self.hidden_layers, self.Y.shape[0]]
        L = len(layer_sizes)

        for i in range(1, L):
            weights.append(np.random.rand(layer_sizes[i], layer_sizes[i-1]) - 0.5)
            biases.append(np.random.rand(layer_sizes[i], 1) - 0.5) 

        return weights, biases


    def _update_parameters(
        self, 
        weights: list, 
        biases: list, 
        gradients: list, 
        bias_gradients: list
    ) -> tuple:
        for i in range(1, len(weights)):
            weights[i] = weights[i] - self.learning_rate * gradients[i]
            biases[i] = biases[i] - self.learning_rate * bias_gradients[i]

        return weights, biases


    # Returns output (activation from final layer)
    # Returns activations per layer
    # X takes a batch of input data
    # X is taken as a parameter instead of from instance field 
    # as test set validation will need different X.
    def _forward_propagate(self, X: np.ndarray, weights: list, biases: list) -> tuple:
        activations = [X]
        L = len(weights)    # Total number of layers, including input layer
        Z = [0]

        for layer_i in range(1, L):
            z_layer_i = (weights[layer_i] @ activations[layer_i - 1]) + biases[layer_i]
            activations.append(self._activation_function(z_layer_i))
            Z.append(z_layer_i)

        # Adding a softmax layer doesn't change the results much. Need to understand more.
        activations[L-1] = self._softmax(activations[L-1])    
        y = activations[L-1]
        return y, activations, Z


    def _backpropagate(self, weights: list, biases: list) -> tuple:
        L = len(weights)
        n = self.X.shape[1]         # Number of datapoints
        _, activations, Z = self._forward_propagate(self.X, weights, biases)

        # Initializing empty gradients list
        gradients = [np.array([0]) for _ in range(L)]
        bias_gradients = [np.array([0]) for _ in range(L)]

        # Calculation for output layer, L
        delta_a_c = activations[L-1] - self.Y        # The 1 / n normalization is applied at the next steps.

        output_error = delta_a_c * self._activation_function_prime(Z[L-1])
        if self._activation_function_type == ActivationFunctionValues.RELU:
            output_error = delta_a_c

        gradients[L-1] = 1 / n * (output_error @ activations[L-2].T)
        bias_gradients[L-1] = 1 / n * np.sum(output_error, axis = 1, keepdims=True)                   

        delta_l_next = output_error                                         # Error from next layer, delta l+1th layer.
        for l in range(L-2, 0, -1):
            delta_l_current = (weights[l+1].T @ delta_l_next) * self._activation_function_prime(Z[l])

            # Here, delta_l_current and activations[l-1] is for all datapoints. l * n and (l-1) * n.
            # The dot product sums up all the a * delta values for the datapoints. Therefore, the 1 / n is used to average it out.
            gradients[l] = 1 / n * (delta_l_current @ activations[l-1].T)
            bias_gradients[l] = 1 / n * np.sum(delta_l_current, axis = 1, keepdims=True)

            delta_l_next = delta_l_current

        return gradients, bias_gradients
    
    # endregion


    # region Public Methods

    def train_neural_network(
        self,
        on_progress: Callable[[TrainingProgress], None] | None = None
    ) -> None:
        weights, biases = self._initialize_weights()

        # Epoch is 1-indexed for better readability
        for epoch in range(1, self.epochs + 1):
            gradients, bias_gradients = self._backpropagate(weights, biases)
            weights, biases = self._update_parameters(weights, biases, gradients, bias_gradients)

            if epoch % 10 == 0 or epoch == self.epochs:
                a_L, _, _ = self._forward_propagate(self.X, weights, biases)
                predictions = self.get_predictions(a_L)

                if on_progress is not None:
                    on_progress(
                        TrainingProgress(
                            completed_epochs=epoch,
                            total_epochs=self.epochs,
                            accuracy=self.get_accuracy(predictions, np.argmax(self.Y, axis=0)),
                            error=self._calculate_error(a_L, self.Y)
                        )
                    )

        self.parameters = {"weights": weights, "biases": biases}


    def predict(self, testing_X: np.ndarray) -> np.ndarray:
        if self.parameters is None:
            raise RuntimeError("Network has not been trained. Call train_neural_network() before predict().")

        probabilities, _, _ = self._forward_propagate(testing_X, self.parameters["weights"], self.parameters["biases"])
        return probabilities


    def save_weights(self, file_path: str | Path) -> None:
        if self.parameters is None:
            raise RuntimeError("Train the network before saving weights.")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {}
        for i in range(1, len(self.parameters["weights"])):
            arrays[f"weight_{i}"] = self.parameters["weights"][i]
            arrays[f"bias_{i}"] = self.parameters["biases"][i]

        np.savez_compressed(path, **arrays)


    def load_weights(self, file_path: str | Path) -> None:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        layer_sizes = [self.X.shape[0], *self.hidden_layers, self.Y.shape[0]]
        weights = [np.array([0])]
        biases = [np.array([0])]

        with np.load(path, allow_pickle=False) as saved:
            for i in range(1, len(layer_sizes)):
                weight = saved[f"weight_{i}"]
                bias = saved[f"bias_{i}"]

                expected_weight_shape = (layer_sizes[i], layer_sizes[i - 1])
                expected_bias_shape = (layer_sizes[i], 1)

                if weight.shape != expected_weight_shape:
                    raise ValueError(
                        f"Weight {i} has shape {weight.shape}; expected {expected_weight_shape}."
                    )
                if bias.shape != expected_bias_shape:
                    raise ValueError(
                        f"Bias {i} has shape {bias.shape}; expected {expected_bias_shape}."
                    )

                weights.append(weight.copy())
                biases.append(bias.copy())

        self.parameters = {"weights": weights, "biases": biases}

    # endregion