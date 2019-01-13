from GA import GA
import numpy

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


if __name__ == '__main__':
    path = "../data/input1.txt"
    inputs, weights, capacity = load_inputs(path)
    ga = GA(inputs, weights, capacity)
    winner = ga.run()
    print("Resenje:", [inputs[i] for i, bit in enumerate(winner) if bit])
    print(winner)   # ovo brisemo posle
    print("Ukupna vrednost: ", numpy.sum(winner * inputs, axis=0))
