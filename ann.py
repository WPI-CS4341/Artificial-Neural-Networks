import sys
import numpy as np

class Artificial_Neural_Network:

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size #num of node of input layer
        self.hidden_layer_size = hidden_layer_size #num of node of hidden layers
        self.output_layer_size = output_layer_size #num of nodes of output layers

        np.random.seed(1)

        #Initialize network
        # self.layer_sizes = [] #
        # self.layer_sizes.append(self.input_layer_size)

        # #Generate layer size array
        # for i in range(self.num_hidden_layers):
        #     self.layer_sizes.append(self.hidden_layer_size)
        #
        # self.layer_sizes.append(self.output_layer_size)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-1 * z))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def classify(self, X):
        hidden_layer = self.sigmoid(np.dot(X, self.weights[0]))
        output = self.sigmoid(np.dot(hidden_layer, self.weights[1]))
        return np.round(output)

    def test(self, test_set, output_set):
        for test in test_set:
            classification = self.classify(test)
            error = output_set - classification
            error_p = np.divide(error, classification)
            print("Test set " + test + " has percent error of " + error_p + "%")

    def train(self, training_set, output_set):
        training_set = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
        output_set = np.array([[0],
            [1],
            [1],
            [0]])
        self.weights = [
            2 * np.random.random((3, 4)) - 1, # W1
            2 * np.random.random((4, 1)) - 1 # W2
        ]
        for n in xrange(self.hidden_layer_size):
            input_layer  = training_set
            hidden_layer = self.sigmoid(np.dot(input_layer, self.weights[0]))
            output_layer = self.sigmoid(np.dot(hidden_layer, self.weights[1]))

            output_layer_error  = output_set - output_layer
            output_layer_delta  = output_layer_error * self.sigmoid_prime(output_layer)

            # print self.weights[1]
            # print output_layer_delta

            hidden_layer_error  = output_layer_delta.dot(self.weights[1])
            hidden_layer_delta  = hidden_layer_error * self.sigmoid_prime(hidden_layer)

            # print hidden_layer_error.shape
            # print hidden_layer_delta.shape

            self.weights[0] += input_set.T.dot(hidden_layer_delta)
            self.weights[1] += hidden_layer.T.dot(output_layer_delta)
