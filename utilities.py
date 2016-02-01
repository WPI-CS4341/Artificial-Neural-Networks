import numpy as np


class Utilities:

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        g = Utilities.sigmoid(z)
        return g * (1 - g)

    @staticmethod
    def scale(x):
        abs_x = np.absolute(x)
        x = x / np.amax(abs_x, axis=0)
        return x

    @staticmethod
    def get_empty_layers(x, network):
        layers = []
        layers.append(np.array(x[:]))

        for i in range(network.num_hidden_layers):
            layers.append([0.0] * network.hidden_layer_size)

        layers.append([0.0] * network.output_layer_size)
        return np.array(layers)
