"""
Authors: Yang Liu (yliu17), Tyler Nickerson (tjnickerson)
Date: Jan 28, 2016
"""

def get_input_matrix(points):
    matrix = []
    for p in points:
        matrix.append([p.x, p.y])
    return matrix

def get_output_matrix(points):
    matrix = []
    for p in points:
        matrix.append(p.label)
    return matrix

def parse_file(filename):
    """Parse the file line to nodes"""
    nodes = []
    with open(filename, "r") as infile:
        for line in infile:
            line = line[:-2]
            data = line.split(" ")
            nodes.append(Point(data[0], data[1], data[2]))
        return nodes

def main():
    args = sys.argv[1:]

    if len(args) > 0:
        filename = args[0]
        num_node = 1
        percentage = 20

        if len(args) > 3:
            if args[1] == "h":
                num_node = int(args[2])
                if args[3] == "p":
                    percentage = float(args[4])
            elif args[1] == "p":
                percentage = float(args[2])

        points = parse_file(filename)
        input_set = get_input_matrix(points)
        output_set = get_output_matrix(points)
        ann = ArtificialNeuralNetwork(num_node)
        ann.training(input_set, output_set)

if __name__ == "__main__":
    main()
