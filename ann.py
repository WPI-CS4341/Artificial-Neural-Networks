import sys
from node import Node


def parse_file(filename):
    """Parse the file line to nodes"""
    nodes = []
    with open(filename, "r") as infile:
        for line in infile:
            line = line[:-2]
            data = line.split(" ")
            nodes.append(Node(data[0], data[1], data[2]))
        return nodes


def main():
    args = sys.argv[1:]
    if len(args) > 0:
        filename = args[0]
        num_node = 0
        percentage = 0
        if len(args) > 3:
            if args[1] == "h":
                num_node = int(args[2])
                if args[3] == "p":
                    percentage = float(args[4])
            elif args[1] == "p":
                percentage = float(args[2])

        nodes = parse_file(filename)


if __name__ == "__main__":
    main()
