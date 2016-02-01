"""
Authors: Yang Liu (yliu17), Tyler Nickerson (tjnickerson)
Date: Jan 28, 2016
"""
import sys
import numpy as np
from ann1 import Artificial_Neural_Network
from bpl import Back_Prop_Learner
from validator import Validator


def parse_file(filename):
    """Parse the file line to nodes"""

    try:
        with open(filename, "r") as infile:
            x = []
            y = []
            for line in infile:
                line = line[:-2]
                data = line.split(" ")
                x.append([float(data[0]), float(data[1])])
                y.append([float(data[2])])
            return x, y
    except IOError as e:
        print(e.errno, e.strerror)


def main():
    args = sys.argv[1:]
    if len(args) > 0:
        filename = args[0]
        hidden_layer_size = 5
        percentage = 0.2
        if len(args) > 2:
            if args[1] == "h":
                hidden_layer_size = int(args[2])
                if len(args) > 3:
                    if args[3] == "p":
                        percentage = float(args[4]) / 100.0
            elif args[1] == "p":
                percentage = float(args[2]) / 100.0

        # Parse file into data sets
        x, y = parse_file(filename)

        # Calculate data set size
        test_set_size = int(len(x) * percentage)
        train_set_size = len(x) - test_set_size

        # Generate training set
        train_set_x = x[:train_set_size]
        train_set_y = y[:train_set_size]

        # Generate test set
        test_set_x = x[train_set_size + 1:]
        test_set_y = y[train_set_size + 1:]

        # Initialize artificial neural network
        ann = Artificial_Neural_Network(
            len(x[0]), 1, hidden_layer_size, len(y[0]))

        # Learn from data
        learner = Back_Prop_Learner()
        learner.learn(train_set_x, train_set_y, ann)

        # Validate the network with test set
        validator = Validator()
        error_rate = validator.validate(test_set_x, test_set_y, ann)

        print("Error rate: {0}".format(error_rate))
if __name__ == "__main__":
    main()
