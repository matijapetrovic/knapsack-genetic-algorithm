import numpy


class Knapsack:
    def __init__(self, path):
        self.inputs = []
        self.weights = []
        self.capacity = 0
        self.load_inputs(path)

    def is_valid_knapsack(self, individual):
        return numpy.sum(individual * self.weights, axis=0) <= self.capacity

    def load_inputs(self, path):
        infile = open(path, "r")
        self.capacity = eval(infile.readline().strip().split(": ")[1])
        for line in infile:
            input, weight = line.strip().split(",")
            self.inputs.append(eval(input))
            self.weights.append(eval(weight))
