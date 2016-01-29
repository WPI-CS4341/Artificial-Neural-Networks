import sys
import numpy
from point import Point

class ArtificialNeuralNetwork:

    def __init__(self, num_hidden_layers):
        self.num_hidden_layers = num_hidden_layers
        self.weights = np.random.randrom(2, 1);

    def sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))

    def sigmoidPrime(x):
        return x*(1-x)

    def classify(point):
        return round(sigmod(np.dot(point, self.weights)))

    def train(self, input_set, output_set):
        percentage = round(len(input_set) * 0.20)
        test_set = input_set[:percentage]
        training_set = input_set[percentage:]
        print training_set
        for n in xrange(self.num_hidden_layers):
            hidden_layer = sigmod(np.dot(training_set, weights))
            error = output_set - hidden_layer
            delta = error * sigmoidPrime()
            self.weights += np.dot(training_set, self.weights)
        print hidden_layer
