"""
Authors: Yang Liu (yliu17), Tyler Nickerson (tjnickerson)
Date: Jan 28, 2016
"""
import sys
import numpy as np
from ann import Artificial_Neural_Network
# from bpl import Back_Prop_Learner

def parse_file(filename):
    """Parse the file line to nodes"""
    examples = []
    outputs = []

    try:
        with open(filename, "r") as infile:
            for line in infile:
                line = line[:-2]
                data = line.split(" ")
                examples.append([float(data[0]), float(data[1])])
                outputs.append([float(data[2])])
    except IOError as e:
        print(e.errno, e.strerror)

    return np.array(examples), np.array(outputs)

def main():
    args = sys.argv[1:]
    if len(args) > 0:
        filename = args[0]
        hidden_layer_size = 5
        percentage = 0.20
        if len(args) > 2:
            if args[1] == "h":
                hidden_layer_size = int(args[2])
                if len(args) > 3:
                    if args[3] == "p":
                        percentage = float(args[4])
            elif args[1] == "p":
                percentage = float(args[2])

        examples, outputs = parse_file(filename)

        percentage      = int(round(len(examples) * percentage))
        testing         = examples[:percentage]
        testing_output  = outputs[:percentage]
        training        = examples[percentage:]
        training_output = outputs[percentage:]

        ann = Artificial_Neural_Network(
            len(training[0]),
            hidden_layer_size,
            len(training_output[0])
        )

        ann.train(training, training_output)
        ann.test(testing, testing_output)

        # learner = Back_Prop_Learner()
        # learner.learn(examples, ann)


if __name__ == "__main__":
    main()
