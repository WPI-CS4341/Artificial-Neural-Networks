import numpy as np
from utilities import Utilities as U


class Back_Prop_Learner:

    def learn(self, x, y, network):
        x = U.scale(x)
        deltas = []
        z = []

        for size in network.layer_sizes[1:]:
            deltas.append(np.array([0.0] * size))

        # Propegate forwards
        layers = U.get_empty_layers(x, network)
        for l in range(0, len(network.layer_sizes) - 1):
            z.append(np.array(layers[l]).dot(np.array(network.weights[l])))
            layers[
                l + 1] = np.round(U.sigmoid(z[l]))

        # Calculate output layer errors
        output_layer_index = len(network.layer_sizes) - 1
        output_error_index = output_layer_index - 1

        # print(layers[output_layer_index])
        deltas[output_error_index] = np.multiply(-(y - layers[output_layer_index]), U.sigmoid_prime(
            np.array(z[output_error_index])))

        # Back propegate errors
        for l in range(output_error_index - 1, -1, -1):
            deltas[l] = np.array(deltas[l + 1]).dot(network.weights[l + 1].T) * U.sigmoid_prime(z[l + 1])

        # Adjust all the weights
        for l in range(output_layer_index):
            network.weights[l] += 2 * layers[l].T.dot(deltas[l])

        return network
