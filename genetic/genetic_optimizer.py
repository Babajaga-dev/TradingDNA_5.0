import random

class GeneticOptimizer:
    def __init__(self, config):
        self.generations = config['genetic']['generations']
        self.population_size = config['genetic']['population_size']
        self.mutation_rate = config['genetic']['mutation_rate']

    def evolve(self, simulator, market_data, gene_classes):
        population = [self._initialize_genes(gene_classes) for _ in range(self.population_size)]
        for generation in range(self.generations):
            fitness = []
            for individual in population:
                signals = self._combine_signals(market_data, individual)
                pnl, final_equity = simulator.run_simulation(market_data, signals)
                fitness.append(self._calculate_fitness(pnl, final_equity))
            sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)]
            population = self._breed(sorted_population[:len(sorted_population)//2])
        return sorted_population[0]

    def _initialize_genes(self, gene_classes):
        return {gene.__name__: gene().apply for gene in gene_classes}

    def _combine_signals(self, data, genes):
        signals = sum(gene(data) for gene in genes.values())
        return signals.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    def _calculate_fitness(self, pnl, final_equity):
        return sum(pnl) / (abs(min(pnl)) + 1)  # Sharpe-like ratio

    def _breed(self, parents):
        offspring = []
        for _ in range(len(parents) * 2):
            parent1, parent2 = random.sample(parents, 2)
            child = self._crossover(parent1, parent2)
            if random.random() < self.mutation_rate:
                self._mutate(child)
            offspring.append(child)
        return offspring

    def _crossover(self, parent1, parent2):
        return {k: random.choice([v, parent2[k]]) for k, v in parent1.items()}

    def _mutate(self, individual):
        for key in individual.keys():
            if random.random() < self.mutation_rate:
                individual[key] += random.uniform(-0.1, 0.1)
