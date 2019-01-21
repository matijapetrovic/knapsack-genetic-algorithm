import numpy
from GA import GA
from Knapsack import Knapsack

if __name__ == '__main__':
    path = "../data/input1.txt"
    ks = Knapsack(path)
    ga = GA(ks)
    winner = ga.run()
    print("GENETSKI ALGORITAM")
    print("===================")
    print("Resenje:", [ks.inputs[i] for i, bit in enumerate(winner) if bit])
    print("Bitna reprezentacija: ", winner)
    print("Ukupna vrednost: ", numpy.sum(winner * ks.inputs, axis=0))
    print("Kapacitet ranca: ", ks.capacity)
    print("Ukupna tezina: ", numpy.sum(winner * ks.weights, axis=0))