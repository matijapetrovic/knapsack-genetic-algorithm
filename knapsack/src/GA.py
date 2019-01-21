import numpy


class GA:
    max_gens = 500
    pop_size = 100
    tol = 50
    num_elites = 1
    num_offspring = pop_size - num_elites
    prob_mutation = 5
    prob_crossover = 80

    def __init__(self, inputs, weights, capacity):
        # npr [[0 1 0 1 0 1 0], [1 - 1 -1= 120012}]]]
        self.population = []
        self.fitness = None
        # npr [6 5 8 9 6 7 3]
        self.inputs = inputs
        self.input_size = len(inputs)
        self.weights = weights
        self.capacity = capacity

    def run(self):
        self.init_population()
        max_fitness = -1
        num_gens_no_change = 0
        for i in range(GA.max_gens):
            # kriterijum zaustavljanja
            if num_gens_no_change >= GA.tol:
                break
            self.next_gen()
            num_gens_no_change = 0 if numpy.max(self.fitness) > max_fitness else num_gens_no_change + 1

        max_fitness_idx = numpy.where(self.fitness == numpy.max(self.fitness))[0][0]
        return self.population[max_fitness_idx, :]

    def init_population(self):
        # inicijalizuj populaciju na niz nizova bitova
        self.population = numpy.array([numpy.random.randint(2, size=self.input_size) for i in range(GA.pop_size)])

    def next_gen(self):
        self.cal_pop_fitness()
        new_population = self.select_mating_pool()
        self.mutate(new_population)

    def cal_pop_fitness(self):
        # mnozi svaki bit sa odgovarajucom vrednoscu predmeta i sabira da bi dobili fitness
        self.fitness = numpy.array([numpy.sum(chromosome * self.inputs, axis=0)
                                   if numpy.sum(chromosome * self.weights) <= self.capacity else -9999999
                                   for chromosome in self.population])

    def select_mating_pool(self):
        # parents ce biti niz num_parents nizova po len(self.inputs) bita
        new_population = numpy.empty((GA.pop_size, self.input_size))
        for i in range(GA.num_elites):
            elite = self.population[numpy.where(self.fitness == numpy.max(self.fitness))[0][0], :]
            new_population[i, :] = elite

        offspring_num = GA.num_elites
        while offspring_num < GA.pop_size - GA.num_elites:
            new_fitness = numpy.multiply(self.fitness, numpy.random.uniform(0, 1, GA.pop_size))
            parent1_idx = numpy.where(self.fitness == numpy.max(self.fitness))[0][0]
            new_fitness[parent1_idx] = -99999999
            parent2_idx = numpy.where(self.fitness == numpy.max(self.fitness))[0][0]
            if numpy.random.randint(0, 100, 1) < GA.prob_crossover:
                new_population[offspring_num, :], new_population[offspring_num + 1, :] = self.crossover(
                    self.population[parent1_idx], self.population[parent2_idx])
                offspring_num += 2
            else:
                new_population[offspring_num, :] = self.population[parent1_idx, :]
                offspring_num += 1
                new_population[offspring_num, :] = self.population[parent2_idx, :]

        return new_population

    def crossover(self, parent1, parent2):
        crossover_point = numpy.uint8(self.input_size / 2)
        offspring1 = numpy.empty(self.input_size)
        offspring2 = numpy.empty(self.input_size)
        offspring1[0:crossover_point] = parent1[0:crossover_point]
        offspring1[crossover_point:] = parent2[crossover_point:]
        offspring2[0:crossover_point] = parent2[0:crossover_point]
        offspring2[crossover_point:] = parent1[crossover_point:]

        return offspring1, offspring2

    def mutate(self, population):
        for idx in range(GA.num_offspring):
            # biramo random gen i invertujemo ga za 10% populacije
            if numpy.random.randint(0, 100, 1) < GA.prob_mutation:
                mutation_gene_idx = numpy.random.randint(0, self.input_size, 1)
                population[idx, mutation_gene_idx] = 0 if population[idx, mutation_gene_idx] else 1

        return population
