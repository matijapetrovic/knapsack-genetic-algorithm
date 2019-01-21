import numpy


class GA:
    max_gens = 200
    pop_size = 100
    tol = 50
    prob_crossover = 0.8
    num_elites = 2
    num_crossover = int((pop_size - num_elites) * prob_crossover)
    num_mutation = pop_size - num_crossover - num_elites

    def __init__(self, knapsack):
        self.knapsack = knapsack
        self.population = None
        self.fitness = None
        self.num_genes = len(knapsack.inputs)

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
        # Inicijalizujemo populaciju pop_size velicine
        self.population = numpy.array([self.init_individual() for i in range(GA.pop_size)])

    def init_individual(self):
        individual = numpy.zeros(self.num_genes, dtype=int)
        # Ubacujemo predmete u ranac dok nam ne predje kapacitet
        # i vadimo poslednji ubacen predmet
        while self.knapsack.is_valid_knapsack(individual):
            gene_idx = numpy.random.randint(self.num_genes, size=1)
            individual[gene_idx] = 1
        individual[gene_idx] = 0

        return individual

    def next_gen(self):
        self.cal_pop_fitness()
        self.population = self.create_new_generation()

    def cal_pop_fitness(self):
        # Sumiramo izmnozene bitove u hromozomu sa vrednostima predemeta
        # i proveravamo da li zadovoljava ogranicenje
        self.fitness = numpy.array([numpy.sum(chromosome * self.knapsack.inputs, axis=0)
                                    if numpy.sum(chromosome * self.knapsack.weights) <= self.knapsack.capacity
                                    else -9999999 for chromosome in self.population])

    def create_new_generation(self):
        new_population = numpy.zeros((GA.pop_size, self.num_genes), dtype=int)
        # Prvo prepisujemo elite iz prosle generacije
        new_population[:GA.num_elites, :] = self.select_elites(numpy.copy(self.fitness))

        # Ukrstamo jedinke num_crossover puta
        for i in range(GA.num_elites, GA.num_crossover):
            new_population[i, :] = self.crossover(self.select_parents_crossover(numpy.copy(self.fitness)))

        # Mutiramo jedinku num_mutation puta
        for i in range(GA.num_elites + GA.num_crossover, GA.num_mutation):
            new_population[i, :] = self.mutate(self.select_parent_mutation(numpy.copy(self.fitness)))

        return new_population

    def crossover(self, parents):
        # Uzimamo random gen iz hromozoma jedinke i sve gene do njega
        # uzimamo od prvog roditelja, a sve gene posle od drugog
        crossover_point = numpy.random.randint(self.num_genes, size=1)[0]
        offspring = numpy.empty(self.num_genes, dtype=int)
        offspring[0:crossover_point] = parents[0][0:crossover_point]
        offspring[crossover_point:] = parents[1][crossover_point:]

        # dodaj da proverava ogranicenje

        return offspring

    def mutate(self, individual):
        # Uzimamo random gen iz hromozoma jedinke i flipujemo ga
        gene_idx = numpy.random.randint(self.num_genes, size=1)
        individual[gene_idx] = 0 if individual[gene_idx] else 1
        # Ako ne zadovoljava ogranicenje necemo da flipujemo
        while not self.knapsack.is_valid_knapsack(individual):
            individual[gene_idx] = 0 if individual[gene_idx] else 1

        return individual

    def select_elites(self, fitness):
        elites = numpy.zeros((GA.num_elites, self.num_genes), dtype=int)
        for i in range(GA.num_elites):
            # Nalazimo najboljih num_elites jedinki iz prosle generacije
            # i netaknute ih cuvamo za sledecu generaciju
            elite_idx = numpy.where(self.fitness == numpy.max(fitness))[0][0]
            fitness[elite_idx] = -9999999
            elites[i, :] = self.population[elite_idx]

        return elites

    def select_parents_crossover(self, fitness):
        max_fitness = numpy.max(fitness)
        # Biramo dva razlicita roditelja za ukrstanje
        parent1_idx = self.roulette_wheel_selection(fitness, max_fitness)
        parent2_idx = parent1_idx
        while parent2_idx == parent1_idx:
            parent2_idx = self.roulette_wheel_selection(fitness, max_fitness)

        return self.population[parent1_idx], self.population[parent2_idx]

    def select_parent_mutation(self, fitness):
        max_fitness = numpy.max(fitness)
        # Biramo jednog roditelja za mutaciju
        parent_idx = self.roulette_wheel_selection(fitness, max_fitness)

        return self.population[parent_idx]

    @staticmethod
    def roulette_wheel_selection(fitness, max_fitness):
        while True:
            # Rulet selekcija sa "stohastickim prihvatanjem"
            idx = numpy.random.randint(GA.pop_size, size=1, dtype=int)
            if numpy.random.uniform(0, 1) < fitness[idx] / max_fitness:
                return idx[0]


