"""
Authors: Yang Liu (yliu17), Tyler Nickerson (tjnickerson)
Date: Jan 28, 2016
"""
import sys
from ann1 import Artificial_Neural_Network
from bpl import Back_Prop_Learner

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
            return x[:5], y[:5]
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
                        percentage = float(args[4])
            elif args[1] == "p":
                percentage = float(args[2])

        x, y = parse_file(filename)
        ann = Artificial_Neural_Network(len(x[0]),1 , hidden_layer_size, len(y[0]))

        learner = Back_Prop_Learner()
        learner.learn(x, y, ann)


if __name__ == "__main__":
    main()
