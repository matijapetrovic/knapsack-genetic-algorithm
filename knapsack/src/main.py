import numpy
from GA import GA


def load_inputs(path):
    inputs = []
    weights = []

    infile = open(path, "r")
    capacity = eval(infile.readline().strip().split(": ")[1])
    for line in infile:
        input, weight = line.strip().split(",")
        inputs.append(eval(input))
        weights.append(eval(weight))

    return inputs, weights, capacity


def knapsack(capacity, weights, inputs, input_size):
    K = numpy.empty((input_size + 1, capacity + 1))
    for i in range(input_size + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                K[i, w] = 0
            elif weights[i - 1] <= w:
                K[i, w] = max(inputs[i - 1] + K[i - 1, w - weights[i - 1]], K[i - 1, w])
            else:
                K[i, w] = K[i - 1, w]

    return K[input_size, capacity]


if __name__ == '__main__':
    path = "../data/input1.txt"
    inputs, weights, capacity = load_inputs(path)

    ga = GA(inputs, weights, capacity)
    winner = ga.run()
    print("GENETIC ALGORITHM")
    print("===================")
    print("Resenje:", [inputs[i] for i, bit in enumerate(winner) if bit])
    print("Bitna reprezentacija: ", winner)
    print("Ukupna vrednost: ", numpy.sum(winner * inputs, axis=0))
    print("Kapacitet ranca: ", capacity)
    print("Ukupna tezina: ", numpy.sum(winner * weights, axis=0))