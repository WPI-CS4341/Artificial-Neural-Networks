import sys
import numpy as np

class Artificial_Neural_Network:

    def __init__(self, hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size
        self.input_weights = np.random.randn(2, self.hidden_layer_size)
        self.output_weights = np.random.randn(self.hidden_layer_size, 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-1 * z))

    # def sigmoidPrime(x):
    #     return x*(1-x)

    def classify(self, X):
        print(x)
        print(self.input_weights)
        print(np.dot(X, self.input_weights))
        # hidden_layer = self.sigmoid(np.dot(X, self.input_weights))
        # output = self.sigmoid(np.dot(hidden_layer, self.output_weights))
        return np.round(output)

    def train(self, input_set, output_set):
        percentage = round(len(training_set) * 0.20)
        test_set = input_set[:percentage]
        training_set = input_set[percentage:]

        for n in xrange(self.num_hidden_layers):
            hidden_layer = self.sigmoid(np.dot(input_set, self.input_weights))
            output_layer = self.sigmoid(np.dot(hidden_layer, self.output_weights))

            output_layer_error = output_set - predicted_output
            output_layer_delta = output_error * self.sigmoidPrime(output_layer)

            hidden_layer_error = output_delta.dot(self.output_weights.T)
            hidden_layer_delta = hidden_layer_error * self.sigmoidPrime(hidden_layer)

            self.input_weights += input_set.T.dot(hidden_layer_delta)
            self.output_weights += hidden_layer.T.dot(output_layer_delta)
