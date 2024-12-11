import numpy as np
import logging
from typing import List, Tuple, Dict, Any

from ..models.genes import (
    TorchGene,
    create_ensemble_gene
)

logger = logging.getLogger(__name__)

class EnsembleGeneticOperator:
    """Gestisce le operazioni genetiche per l'ensemble"""
    
    def __init__(self, mutation_rate: float, tournament_size: int):
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def mutate_ensemble(self, ensemble: List[TorchGene]) -> None:
        """
        Muta ogni gene dell'ensemble
        
        Args:
            ensemble: Ensemble da mutare
        """
        try:
            for gene in ensemble:
                if np.random.random() < self.mutation_rate:
                    gene.mutate(np.random.uniform(0.2, 0.8))
        except Exception as e:
            logger.error(f"Error mutating ensemble: {e}")

    def crossover_ensembles(self, parent1: List[TorchGene], parent2: List[TorchGene]) -> List[TorchGene]:
        """
        Crossover tra due ensemble
        
        Args:
            parent1: Primo ensemble genitore
            parent2: Secondo ensemble genitore
            
        Returns:
            Nuovo ensemble figlio
        """
        try:
            child_ensemble: List[TorchGene] = []
            
            for gene1, gene2 in zip(parent1, parent2):
                child_gene = gene1.__class__(random_init=False)
                child_dna: Dict[str, Any] = {}
                
                for key in set(gene1.dna.keys()) | set(gene2.dna.keys()):
                    if np.random.random() < 0.5:
                        child_dna[key] = gene1.dna.get(key, gene2.dna[key])
                    else:
                        child_dna[key] = gene2.dna.get(key, gene1.dna[key])
                    
                    if isinstance(child_dna[key], (int, float)):
                        variation = np.random.uniform(-0.1, 0.1)
                        child_dna[key] *= (1.0 + variation)
                
                child_gene.dna = child_dna
                child_ensemble.append(child_gene)
            
            return child_ensemble
            
        except Exception as e:
            logger.error(f"Error in crossover: {e}")
            return create_ensemble_gene(random_init=True)

    def select_parent(self, evaluated_population: List[Tuple[List[TorchGene], float]]) -> List[TorchGene]:
        """
        Seleziona genitore usando tournament selection
        
        Args:
            evaluated_population: Lista di tuple (ensemble, fitness)
            
        Returns:
            Ensemble selezionato
        """
        try:
            tournament = np.random.choice(len(evaluated_population), size=self.tournament_size, replace=False)
            tournament_pop = [evaluated_population[i] for i in tournament]
            
            weights = [max(0.1, score) for _, score in tournament_pop]
            weights = np.array(weights) / sum(weights)
            
            selected_idx = np.random.choice(len(tournament_pop), p=weights)
            return tournament_pop[selected_idx][0]
            
        except Exception as e:
            logger.error(f"Error selecting parent: {e}")
            return create_ensemble_gene(random_init=True)

    def create_next_generation(self, 
                             evaluated_population: List[Tuple[List[TorchGene], float]],
                             population_size: int,
                             elite_size: int) -> List[List[TorchGene]]:
        """
        Crea la prossima generazione di ensemble
        
        Args:
            evaluated_population: Lista di tuple (ensemble, fitness)
            population_size: Dimensione della popolazione
            elite_size: Numero di elite da preservare
            
        Returns:
            Nuova popolazione
        """
        try:
            sorted_population = sorted(evaluated_population, key=lambda x: x[1], reverse=True)
            new_population = [ensemble.copy() for ensemble, _ in sorted_population[:elite_size]]
            
            while len(new_population) < population_size:
                if np.random.random() < 0.8:  # 80% crossover
                    parent1 = self.select_parent(sorted_population)
                    parent2 = self.select_parent(sorted_population)
                    child = self.crossover_ensembles(parent1, parent2)
                    self.mutate_ensemble(child)
                else:
                    child = create_ensemble_gene(random_init=True)
                new_population.append(child)
            
            return new_population
            
        except Exception as e:
            logger.error(f"Error creating next generation: {e}")
            return [create_ensemble_gene(random_init=True) for _ in range(population_size)]

    def create_initial_population(self, population_size: int) -> List[List[TorchGene]]:
        """
        Crea popolazione iniziale di ensemble
        
        Args:
            population_size: Dimensione della popolazione
            
        Returns:
            Lista di ensemble
        """
        try:
            return [create_ensemble_gene(random_init=True) 
                    for _ in range(population_size)]
        except Exception as e:
            logger.error(f"Error creating initial population: {e}")
            return []
