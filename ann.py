import sys
import numpy as np

class Artificial_Neural_Network(object):

    debug = 1

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size #num of node of input layer
        self.hidden_layer_size = hidden_layer_size #num of node of hidden layers
        self.output_layer_size = output_layer_size #num of nodes of output layers

        np.random.seed(1)

        #Initialize network
        # self.layer_sizes = [] #
        # self.layer_sizes.append(self.input_layer_size)

        self.weights = [
            1 * np.random.random((self.input_layer_size, self.hidden_layer_size)) - 1, # W1
            1 * np.random.random((self.hidden_layer_size, self.output_layer_size)) - 1 # W2
        ]

        # #Generate layer size array
        # for i in range(self.num_hidden_layers):
        #     self.layer_sizes.append(self.hidden_layer_size)
        #
        # self.layer_sizes.append(self.output_layer_size)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def classify(self, X):
        hidden_layer = self.sigmoid(np.dot(X, self.weights[0]))
        output = self.sigmoid(np.dot(hidden_layer, self.weights[1]))
        return np.round(output)

    def test(self, test_set, output_set):
        classification = self.classify(test_set)
        error = output_set - classification
        percent_error = float(np.count_nonzero(error)) / float(len(test_set))
        print("Test set has percent error of " + str(percent_error * 100) + "%")

    def train(self, training_set, output_set):
        overall_error = 1
        timeout = np.power(self.hidden_layer_size, 4)
        i = 0
        while (i < timeout) :
            input_layer  = training_set
            hidden_layer = self.sigmoid(np.dot(input_layer, self.weights[0]))
            output_layer = self.sigmoid(np.dot(hidden_layer, self.weights[1]))

            output_layer_error  = output_set - output_layer
            output_layer_delta  = output_layer_error * self.sigmoid_prime(output_layer)

            hidden_layer_error  = output_layer_delta.dot(self.weights[1].T)
            hidden_layer_delta  = hidden_layer_error * self.sigmoid_prime(hidden_layer)

            self.weights[0] += input_layer.T.dot(hidden_layer_delta)
            self.weights[1] += hidden_layer.T.dot(output_layer_delta)

            total_wrong         = np.count_nonzero(output_set - np.round(output_layer))
            total_percent_error = float(total_wrong) / float(len(output_layer))

            if total_percent_error < overall_error:
                overall_error = total_percent_error
                i = 0
            else:
                i += 1

            if self.debug:
                print "Computed percent error " + str(overall_error) + " " + str(i) + " Threshold not reached."

        if self.debug:
            print "Training ended with percent error " + str(overall_error) + "."
