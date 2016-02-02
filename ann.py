"""
Authors: Yang Liu (yliu17), Tyler Nickerson (tjnickerson)
Date: Jan 28, 2016
"""
import sys
import os.path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

DEBUG = 0


class Artificial_Neural_Network(object):
    """The artifical neural network, integrated with training and testing"""
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        """Initialize network attributes"""
        # Number of node of input layer
        self.input_layer_size = input_layer_size
        # Number of node of hidden layers
        self.hidden_layer_size = hidden_layer_size
        # Number of nodes of output layers
        self.output_layer_size = output_layer_size

        # Initialize random number generator
        np.random.seed(1)

        # Initilize  weights to random numbers
        self.weights = [
            # Weight between input and hidden layer, a (2 X hidden_layer_size)
            # matrix
            2 * np.random.random((self.input_layer_size,
                                  self.hidden_layer_size)) - 1,
            # Weight between hidden and output layer, a (hidden_layer_size X 1)
            # matrix
            2 * np.random.random((self.hidden_layer_size,
                                  self.output_layer_size)) - 1
        ]

    def sigmoid(self, z):
        """Logistic activation function"""
        # g(a), a = x dot weights
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, g):
        """Gradient of activation function"""
        # For logistic function, dg/dz = g(z) x (1 - g(z))
        return g * (1 - g)

    def classify(self, X):
        """Forward propagate through the network and classify based on inputs"""
        # a2 = g(z1), z1 = X dot weight1
        hidden_layer = self.sigmoid(np.dot(X, self.weights[0]))
        # a3 = g(z2), z2 = a2 dot weight2
        output = self.sigmoid(np.dot(hidden_layer, self.weights[1]))
        # Round the outputs to 0 or 1
        return np.round(output)

    def test(self, test_set, output_set, graph=False):
        """Test the artifical neural network"""
        # Classify the test set
        classification = self.classify(test_set)
        # Error = y - yHat
        error = output_set - classification
        # No error when error is 0, percent error = num of error / size of test set
        percent_error = float(np.count_nonzero(error)) / float(len(test_set))
        print("Test set has percent error of {0}%".format(percent_error * 100))
        # Plot the results
        if graph:
            # Concatinate error column with test inputs
            point_errors = np.hstack((test_set, error))
            # Filter out the correct inputs
            correct = np.array(filter(lambda line: line[2] == 0, point_errors))
            # Filter out the incorrect inputs
            incorrect = np.array(filter(lambda line: line[2] != 0, point_errors))

            # Seperate inputs x1 and x2
            correct_x1 = correct[:, 0]
            correct_x2 = correct[:, 1]
            incorrect_x1 = incorrect[:, 0]
            incorrect_x2 = incorrect[:, 1]

            # Green dots for correct predctions
            green_dot, = plt.plot(correct_x1, correct_x2, "go", markersize=6)
            # Red dots for incorrect predictions
            red_dot, = plt.plot(incorrect_x1, incorrect_x2, "ro", markersize=6)
            # Add labels to plots
            plt.legend([green_dot, (green_dot, red_dot)], ["Correct", "Wrong"])
            # Display the plot
            plt.show()

    def train(self, training_set, output_set):
        """Train the network with backward propagation"""
        # number errors occured during training
        overall_error = 1
        # Stop training when time out
        timeout = np.power(self.hidden_layer_size, 4)
        # Counter
        i = 0
        # Weights generating the lowest error rates
        lowest_weights = []
        while (i < timeout):
            # Propagate forward through the network
            input_layer = training_set
            # a1 = g(X dot weight1)
            hidden_layer = self.sigmoid(np.dot(input_layer, self.weights[0]))
            # a2 = g(a1 dot weight2)
            output_layer = self.sigmoid(np.dot(hidden_layer, self.weights[1]))

            # Error3 = y - yHat
            output_layer_error = output_set - output_layer
            # Delta3 = error3 x g'(a3)
            output_layer_delta = output_layer_error * \
                self.sigmoid_prime(output_layer)

            # Error2 = delta3 dot (weight2)T
            hidden_layer_error = output_layer_delta.dot(self.weights[1].T)
            # Delta2 = error2 x g'(a2)
            hidden_layer_delta = hidden_layer_error * \
                self.sigmoid_prime(hidden_layer)

            # Weight1 = weight1 + (X)T dot delta2
            self.weights[0] += input_layer.T.dot(hidden_layer_delta)
            # Weight2 = weight2 + (in2)T dot delta3
            self.weights[1] += hidden_layer.T.dot(output_layer_delta)

            # total error = y - yHat
            total_wrong = np.count_nonzero(output_set - np.round(output_layer))
            # No error when error is 0, percent error = num of error / size of test set
            total_percent_error = float(total_wrong) / float(len(output_layer))

            # Find the weight with minimum percentage of error
            if total_percent_error < overall_error:
                overall_error = total_percent_error
                i = 0
                lowest_weights = self.weights
            else:
                i += 1

            # Debug message
            if DEBUG:
                print("Computed percent error {0} {1}. Error threshold not met.".format(
                    overall_error, i))

        # Keep minimum weights for the network
        self.weights = lowest_weights

        # Debug message
        if DEBUG:
            print("Training ended with percent error {0}.".format(
                overall_error))


def parse_file(filename):
    """Parse the file line to nodes"""
    # Hold parsed inputs
    examples = []
    # Hold parsed outputs
    outputs = []

    # Read each line and add to the examples and output lists
    if os.path.isfile(filename):
        with open(filename, "r") as infile:
            for line in infile:
                # Remove new line character and carriage return
                line = line[:-2]
                # Split data into string list
                data = line.split(" ")

                # Add to list as floating point numbers
                examples.append([float(data[0]), float(data[1])])
                outputs.append([float(data[2])])
    else:
        # Throw error when cannot open file
        print("Input file does not exist.")

    # Return the inputs and outputs
    return np.array(examples), np.array(outputs)


def main():
    # Read command line arguments
    args = sys.argv[1:]
    # More than 1 argument supplied
    if len(args) > 0:
        # Get data filename
        filename = args[0]
        # Set default number of hidden nodes to 5
        hidden_layer_size = 5
        # Set default hold out data to 20%
        percentage = 0.20
        # When node number or supplied
        if len(args) > 3:
            if args[1] == "h":
                # Set up number of nodes in the hidden layer
                hidden_layer_size = int(args[2])
                if len(args) > 3:
                    if args[3] == "p":
                        # Setup hold out percentage
                        percentage = float(args[4])
            elif args[1] == "p":
                # Setup hold out percentage
                percentage = float(args[2])

        # Read data into memory
        examples, outputs = parse_file(filename)

        # Train only when there is data
        if (examples.any() and outputs.any()):
            # Split out the training set and test set
            percentage = int(round(len(examples) * percentage))
            testing = examples[:percentage]
            testing_output = outputs[:percentage]
            training = examples[percentage:]
            training_output = outputs[percentage:]

            # Initilize instance of the network
            ann = Artificial_Neural_Network(
                len(training[0]),
                hidden_layer_size,
                len(training_output[0])
            )

            # Train the network
            ann.train(training, training_output)
            # Test the network
            ann.test(testing, testing_output, True)
        else:
            # Do nothing when no data supplied
            print("No data to train and test the network")
    else:
        # Show usage when not providing enough argument
        print("Usage: python ann.py <filename> [h <number of hidden nodes> | p <holdout percentage>]")

if __name__ == "__main__":
    main()
