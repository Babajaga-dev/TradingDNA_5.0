# src/optimization/genetic_population.py
import logging
from typing import List, Set, Tuple, Dict, Any, Optional
from copy import deepcopy
import numpy as np
from ..models.genes.base import TradingGene

logger = logging.getLogger(__name__)

class PopulationManager:
    """Gestisce la popolazione genetica"""
    
    def __init__(self, population_size: int, mutation_rate: float, elite_fraction: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[TradingGene] = []
        self.elite_size = max(1, int(population_size * elite_fraction))

    def initialize_population(self) -> List[TradingGene]:
        """
        Inizializza popolazione con diversità garantita
        
        Returns:
            Lista di geni iniziali
        """
        try:
            population: List[TradingGene] = []
            unique_signatures: Set[Tuple] = set()
            
            while len(population) < self.population_size:
                gene = TradingGene(random_init=True)
                signature = self._get_gene_signature(gene)

                if signature not in unique_signatures:
                    unique_signatures.add(signature)
                    population.append(gene)

            self.population = population
            return population
            
        except Exception as e:
            logger.error(f"Error initializing population: {e}")
            self.population = [TradingGene(random_init=True) 
                             for _ in range(self.population_size)]
            return self.population

    def _get_gene_signature(self, gene: TradingGene) -> Tuple:
        """
        Genera signature unica per gene
        
        Args:
            gene: Gene da analizzare
            
        Returns:
            Tuple rappresentante la signature del gene
        """
        try:
            signature = []
            for key, value in sorted(gene.dna.items()):
                if isinstance(value, dict):
                    dict_items = []
                    for k, v in sorted(value.items()):
                        if isinstance(v, (int, float)):
                            v = round(v, 2)
                        dict_items.append((k, v))
                    signature.append((key, tuple(dict_items)))
                elif isinstance(value, (int, float)):
                    signature.append((key, round(value, 2)))
                else:
                    signature.append((key, value))
            return tuple(signature)
            
        except Exception as e:
            logger.error(f"Error generating gene signature: {e}")
            return tuple()

    def calculate_diversity(self) -> float:
        """
        Calcola la diversità della popolazione
        
        Returns:
            Score di diversità tra 0 e 1
        """
        if not self.population:
            return 0.0
            
        try:
            unique_signatures: Set[Tuple] = set()
            total_genes = len(self.population)
            
            fitness_scores = [gene.fitness_score or 0 for gene in self.population]
            if fitness_scores:
                min_fitness = min(fitness_scores)
                max_fitness = max(fitness_scores)
                fitness_range = max_fitness - min_fitness if max_fitness > min_fitness else 1
                
            weighted_signatures = []
            for gene in self.population:
                signature = []
                normalized_fitness = ((gene.fitness_score or 0) - min_fitness) / fitness_range if fitness_range > 0 else 0
                fitness_weight = 1.0 + normalized_fitness
                
                for key, value in sorted(gene.dna.items()):
                    if isinstance(value, dict):
                        dict_items = []
                        for k, v in sorted(value.items()):
                            if isinstance(v, (int, float)):
                                weighted_v = v * fitness_weight
                                discrete_v = round(weighted_v * 100) / 100
                                dict_items.append((k, discrete_v))
                            else:
                                dict_items.append((k, v))
                        signature.append((key, tuple(dict_items)))
                    elif isinstance(value, (int, float)):
                        weighted_v = value * fitness_weight
                        discrete_v = round(weighted_v * 100) / 100
                        signature.append((key, discrete_v))
                    else:
                        signature.append((key, value))
                
                weighted_signatures.append(tuple(signature))
            
            unique_signatures = set(weighted_signatures)
            simple_diversity = len(unique_signatures) / total_genes
            
            performance_bonus = 0.0
            if fitness_scores:
                top_performers = len([s for s in fitness_scores if s > (max_fitness * 0.8)])
                performance_bonus = min(0.1, (top_performers / total_genes) * 0.2)
            
            final_diversity = simple_diversity + performance_bonus
            
            return min(1.0, final_diversity)
            
        except Exception as e:
            logger.error(f"Error calculating diversity: {e}")
            return 0.0

    def inject_diversity(self) -> None:
        """Inietta diversità nella popolazione"""
        try:
            num_random = int(len(self.population) * 0.2)
            self.population[-num_random:] = [TradingGene(random_init=True) 
                                           for _ in range(num_random)]
        except Exception as e:
            logger.error(f"Error injecting diversity: {e}")

    def get_elite(self) -> List[TradingGene]:
        """
        Ottiene i migliori individui della popolazione
        
        Returns:
            Lista degli individui elite
        """
        try:
            sorted_pop = sorted(self.population, 
                              key=lambda x: x.fitness_score or 0, 
                              reverse=True)
            return deepcopy(sorted_pop[:self.elite_size])
        except Exception as e:
            logger.error(f"Error getting elite: {e}")
            return []

    def perform_restart(self, mutation_multiplier: float = 2.2) -> None:
        """
        Esegue il restart della popolazione
        
        Args:
            mutation_multiplier: Moltiplicatore per il tasso di mutazione
        """
        try:
            elite = self.get_elite()
            new_population = []
            
            # Muta elite
            for gene in elite:
                mutated = deepcopy(gene)
                mutated.mutate(self.mutation_rate * mutation_multiplier)
                new_population.append(mutated)
            
            # Riempi con nuovi individui
            while len(new_population) < self.population_size:
                new_population.append(TradingGene(random_init=True))
            
            self.population = new_population
            
        except Exception as e:
            logger.error(f"Error performing restart: {e}")
            self.population = [TradingGene(random_init=True) 
                             for _ in range(self.population_size)]
