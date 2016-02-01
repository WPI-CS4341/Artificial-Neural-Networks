import numpy as np
from ann1 import Artificial_Neural_Network

class Validator:

    def validate(self, test_set_x, test_set_y, network):
        if len(test_set_x) > 0 and len(test_set_y) and network is not None:
            results = network.classify(test_set_x)
            # print(results)
            # print(test_set_y)
            differences = test_set_y - results
            # print(differences)
            data_size = len(differences)
            num_error = np.count_nonzero(differences)
            return float(num_error) / float(data_size)
        return 0.0
