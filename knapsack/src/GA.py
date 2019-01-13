import numpy


class GA:
    max_gens = 100
    pop_size = 100
    num_parents = pop_size // 2
    num_offspring = pop_size - num_parents

    def __init__(self, inputs, weights, capacity):
        # npr [[0 1 0 1 0 1 0], [1 - 1 -1= 120012}]]]
        self.population = []
        # npr [6 5 8 9 6 7 3]
        self.inputs = inputs
        self.input_size = len(inputs)
        self.weights = weights
        self.capacity = capacity

    def run(self):
        self.init_population()
        for i in range(GA.max_gens):
            self.next_gen()

        # nadji najboljeg i vrati ga
        fitness = numpy.array([numpy.sum(chromosome * self.inputs, axis=0) for chromosome in self.population])
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))[0][0]
        return self.population[max_fitness_idx, :]

    def init_population(self):
        # inicijalizuj populaciju na niz nizova bitova
        self.population = numpy.array([numpy.random.randint(2, size=self.input_size) for i in range(GA.pop_size)])

    def next_gen(self):
        fitness = self.cal_pop_fitness()
        parents = self.select_mating_pool(fitness)
        offspring = self.mutation(self.crossover(parents))
        self.population[0:GA.num_parents, :] = parents
        self.population[GA.num_parents:, :] = offspring

    def cal_pop_fitness(self):
        # mnozi svaki bit sa odgovarajucom vrednoscu predmeta i sabira da bi dobili fitness
        fitness = numpy.array([numpy.sum(chromosome * self.inputs, axis=0) for chromosome in self.population])
        return fitness

    def select_mating_pool(self, fitness):
        # parents ce biti niz num_parents nizova po len(self.inputs) bita
        parents = numpy.empty((GA.num_parents, len(self.inputs)))

        for parent_num, parent in enumerate(parents):
            # nalazi indeks hromozoma sa max fitnessom
            max_fitness_idx = numpy.where(fitness == numpy.max(fitness))[0][0]
            # stavlja ga u niz roditelja
            parents[parent_num, :] = self.population[max_fitness_idx, :]
            # podesavamo da ga ne uzmemo opet
            fitness[max_fitness_idx] = -999999999

        return parents

    def crossover(self, parents):
        offspring = numpy.empty((GA.num_offspring, self.input_size))
        crossover_point = numpy.uint8(self.input_size / 2)

        for k in range(len(offspring)):
            # uzimamo 2 roditelja i mesamo ih u dete
            # promenicemo da ih uzima random!!!
            parent1_idx = k % GA.num_parents
            parent2_idx = (k + 1) % GA.num_parents
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

        return offspring

    def mutation(self, offspring):
        for idx in range(GA.num_offspring):
            # biramo random gen i invertujemo ga
            mutation_gene = numpy.random.randint(0, self.input_size, 1)
            offspring[idx, mutation_gene] = 0 if offspring[idx, mutation_gene] else 1

        return offspring
