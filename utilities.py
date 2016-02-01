import numpy as np
class Utilities:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return Utilities.sigmoid(z) * (1 - Utilities.sigmoid(z))

    @staticmethod
    def scale(x):
        abs_x = np.absolute(x)
        x = x / np.amax(abs_x, axis=0)
        return x
