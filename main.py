"""
Authors: Yang Liu (yliu17), Tyler Nickerson (tjnickerson)
Date: Jan 28, 2016
"""
import sys
from ann import Artificial_Neural_Network

def parse_file(filename):
    """Parse the file line to nodes"""
    x = []
    y = []
    try:
        with open(filename, "r") as infile:
            for line in infile:
                line = line[:-2]
                data = line.split(" ")
                x.append([data[0], data[1]])
                y.append(data[2])
            return x, y
    except IOError as e:
        print(e.errno, e.strerror)

def main():
    args = sys.argv[1:]
    if len(args) > 0:
        filename = args[0]
        hidden_layer_size = 0
        percentage = 0
        if len(args) > 2:
            if args[1] == "h":
                hidden_layer_size = int(args[2])
                if len(args) > 3:
                    if args[3] == "p":
                        percentage = float(args[4])
            elif args[1] == "p":
                percentage = float(args[2])

        inputs, outputs = parse_file(filename)
        ann = Artificial_Neural_Network(hidden_layer_size)
        for x in inputs:
            print(int(ann.classify(x)))


if __name__ == "__main__":
    main()
