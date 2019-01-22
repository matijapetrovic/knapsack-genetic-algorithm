import numpy
from GA import GA
from Knapsack import Knapsack


def run_once(ga):
    winner = ga.run()
    print("GENETSKI ALGORITAM")
    print("===================")
    print("Resenje:", [ks.inputs[i] for i, bit in enumerate(winner) if bit])
    print("Bitna reprezentacija: ", winner)
    print("Ukupna vrednost: ", numpy.sum(winner * ks.inputs, axis=0))
    print("Kapacitet ranca: ", ks.capacity)
    print("Ukupna tezina: ", numpy.sum(winner * ks.weights, axis=0))


def run_multiple(ga, iters):
    sum_results = 0
    max_res = -1
    for i in range(iters):
        winner = ga.run()
        res = numpy.sum(winner * ks.inputs, axis=0)
        if res > max_res:
            max_res = res
        sum_results += res
    print("GENETSKI ALGORITAM")
    print("==================")
    print("Nakon %d pokretanja, prosecna vrednost resenja: %d, a maksimalna: %d" % (iters, sum_results / iters, max_res))


if __name__ == '__main__':
    path = "../data/input1.txt"
    ks = Knapsack(path)
    max_gens = 100
    pop_size = 500
    tol = 50
    prob_crossover = 0.8
    num_elites = 2

    ga = GA(ks, max_gens, pop_size, tol, prob_crossover, num_elites)
    run_once(ga)
    run_multiple(ga, 100)
