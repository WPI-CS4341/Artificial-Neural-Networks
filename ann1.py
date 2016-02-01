import sys
import numpy as np
from utilities import Utilities as U


class Artificial_Neural_Network:

    def __init__(self, input_layer_size, num_hidden_layers, hidden_layer_size, output_layer_size):
        self.input_layer_size  = input_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        # Initialize network
        self.layer_sizes = []
        self.layer_sizes.append(self.input_layer_size)
        self.weights = []

        for i in range(self.num_hidden_layers):
            self.layer_sizes.append(self.hidden_layer_size)

        self.layer_sizes.append(self.output_layer_size)
        np.random.seed(1)

        self.init_weights()

    def init_weights(self):
        for i in range(1, len(self.layer_sizes)):
            # Initialize weights to random small number
            w = np.random.randn(self.layer_sizes[i - 1], self.layer_sizes[i])
            self.weights.append(w)

    def forward(self, x):
        inputs = U.scale(x)
        for l in range(1, len(self.layer_sizes)):
            inputs = U.sigmoid(np.array(inputs).dot(self.weights[l - 1]))
        return inputs

    def classify(self, x):
        return np.round(self.forward(x))
