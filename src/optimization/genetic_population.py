# src/optimization/genetic_population.py
import logging
from typing import List, Set, Tuple, Dict, Any, Optional
from copy import deepcopy
import numpy as np
from ..models.genes.base import TradingGene

logger = logging.getLogger(__name__)

class PopulationManager:
    """Gestisce la popolazione genetica"""
    
    def __init__(self, population_size: int, mutation_rate: float, elite_fraction: float = 0.1, config: Dict = None):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[TradingGene] = []
        self.elite_size = max(1, int(population_size * elite_fraction))
        
        # Parametri diversity dal config
        diversity_params = config.get("genetic.diversity", {}) if config else {}
        self.top_performer_threshold = diversity_params.get("top_performer_threshold", 0.8)
        self.performance_bonus_limit = diversity_params.get("performance_bonus_limit", 0.1)
        self.performance_bonus_multiplier = diversity_params.get("performance_bonus_multiplier", 0.2)
        self.injection_fraction = diversity_params.get("injection_fraction", 0.2)
        
        # Validazione parametri
        self._validate_parameters()
        
        logger.info("PopulationManager inizializzato con parametri diversity:")
        logger.info(f"Top performer threshold: {self.top_performer_threshold}")
        logger.info(f"Performance bonus limit: {self.performance_bonus_limit}")
        logger.info(f"Performance bonus multiplier: {self.performance_bonus_multiplier}")
        logger.info(f"Injection fraction: {self.injection_fraction}")

    def _validate_parameters(self):
        """Valida e corregge i parametri se necessario"""
        # Top performer threshold
        if not 0 < self.top_performer_threshold < 1:
            logger.warning(f"Invalid top_performer_threshold: {self.top_performer_threshold}, using default 0.8")
            self.top_performer_threshold = 0.8
            
        # Performance bonus limit
        if not 0 < self.performance_bonus_limit < 1:
            logger.warning(f"Invalid performance_bonus_limit: {self.performance_bonus_limit}, using default 0.1")
            self.performance_bonus_limit = 0.1
            
        # Performance bonus multiplier
        if self.performance_bonus_multiplier <= 0:
            logger.warning(f"Invalid performance_bonus_multiplier: {self.performance_bonus_multiplier}, using default 0.2")
            self.performance_bonus_multiplier = 0.2
            
        # Injection fraction
        if not 0 < self.injection_fraction < 1:
            logger.warning(f"Invalid injection_fraction: {self.injection_fraction}, using default 0.2")
            self.injection_fraction = 0.2

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
            
            # Filtra i valori NaN dai fitness scores
            fitness_scores = [gene.fitness_score for gene in self.population if gene.fitness_score is not None]
            if not fitness_scores:  # Se non ci sono fitness scores validi
                return 0.0
                
            # Calcola min e max fitness escludendo NaN
            min_fitness = np.nanmin(fitness_scores)
            max_fitness = np.nanmax(fitness_scores)
            fitness_range = max_fitness - min_fitness if max_fitness > min_fitness else 1
            
            weighted_signatures = []
            for gene in self.population:
                if gene.fitness_score is None:
                    continue
                    
                signature = []
                normalized_fitness = (gene.fitness_score - min_fitness) / fitness_range if fitness_range > 0 else 0
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
            
            if not weighted_signatures:  # Se non ci sono signatures valide
                return 0.0
                
            unique_signatures = set(weighted_signatures)
            simple_diversity = len(unique_signatures) / total_genes
            
            # Calcola performance bonus usando i parametri configurati
            performance_bonus = 0.0
            if fitness_scores:
                top_performers = len([s for s in fitness_scores if s > (max_fitness * self.top_performer_threshold)])
                performance_bonus = min(
                    self.performance_bonus_limit,
                    (top_performers / total_genes) * self.performance_bonus_multiplier
                )
                
                logger.debug(f"Diversity calculation - Simple: {simple_diversity:.3f}, "
                           f"Performance bonus: {performance_bonus:.3f}")
            
            final_diversity = simple_diversity + performance_bonus
            return min(1.0, final_diversity)
            
        except Exception as e:
            logger.error(f"Error calculating diversity: {e}")
            return 0.0

    def inject_diversity(self) -> None:
        """Inietta diversità nella popolazione"""
        try:
            num_random = int(len(self.population) * self.injection_fraction)
            if num_random < 1:
                logger.warning("Injection fraction too small, using minimum value")
                num_random = 1
                
            logger.info(f"Injecting {num_random} new random individuals")
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
            # Filtra individui con fitness score valido
            valid_population = [gene for gene in self.population if gene.fitness_score is not None]
            if not valid_population:
                return []
                
            sorted_pop = sorted(valid_population, 
                              key=lambda x: x.fitness_score, 
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
            logger.info(f"Population restart completed with {len(new_population)} individuals")
            
        except Exception as e:
            logger.error(f"Error performing restart: {e}")
            self.population = [TradingGene(random_init=True) 
                             for _ in range(self.population_size)]
