import numpy as np
from utilities import Utilities as U


class Back_Prop_Learner:

    def get_empty_layers(self, x, network):
        layers = []
        layers.append(np.array(x[:]))

        for i in range(network.num_hidden_layers):
            layers.append([0.0] * network.hidden_layer_size)

        layers.append([0.0] * network.output_layer_size)
        return np.array(layers)

    def learn(self, x, y, network):
        deltas = []
        for size in network.layer_sizes[1:]:
            deltas.append(np.array([0.0] * size))

        # print(errors)

        # Propegate forwards
        layers = self.get_empty_layers(x, network)
        # print(np.array(layers))
        for l in range(0, len(network.layer_sizes) - 1):
            layers[l + 1] = U.sigmoid(np.array(layers[l]).dot(np.array(network.weights[l])))

        # print(layers)

        # Calculate output layer errors
        output_layer_index = len(network.layer_sizes) - 1
        output_error_index = output_layer_index - 1

        # print(layers[output_layer_index])
        deltas[output_error_index] = np.multiply(
            U.sigmoid_prime(np.array(layers[output_error_index]).dot(np.array(network.weights[output_error_index]))),
            -(y - layers[output_layer_index]))

        # Back propegate errors
        for l in range(output_error_index, 0, -1):
            deltas[l - 1] = U.sigmoid_prime(np.array(layers[l]).dot(np.array(network.weights[l]))) * np.multiply(np.array(deltas[l]), network.weights[l].T)

        #Adjust all the weights
        for l in range(1):
            network.weights[l] += layers[l].T.dot(deltas[l])

        return network
