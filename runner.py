from homemade_neural_network import NeuralNetwork, ActivationFunctionValues

import numpy as np
import pandas as pd

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

    network = NeuralNetwork(
        X = X_train, 
        Y = transformed_y, 
        hidden_layers = [64, 32], 
        epochs = 500, 
        learning_rate = 0.1,
        activation_function_type = ActivationFunctionValues.RELU
    )

    network.train_neural_network(
        on_progress = lambda progress: print(
            f"Epoch: {progress.completed_epochs}/{progress.total_epochs} | "
            f"Accuracy: {progress.accuracy} | Error: {progress.error}"
        )
    )
    # network.save_weights('./data/saved_weights_64_32_mnist.npz')

    # network.load_weights('./data/saved_weights_64_32_mnist.npz')
    # validation_probabilities = network.predict(X_dev)
    # predictions = network.get_predictions(validation_probabilities)
    # print(f'Observed:   {Y_dev}')
    # print(f'Predicted:  {predictions}')
    # print(f'Validation Accuracy: {network.get_accuracy(predictions, Y_dev)}')


if __name__ == '__main__':
    main()