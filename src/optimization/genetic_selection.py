# src/optimization/genetic_selection.py
import logging
from typing import List, Set, Tuple, Dict, Any, Optional
import numpy as np
from copy import deepcopy
from ..models.genes.base import TradingGene

logger = logging.getLogger(__name__)

class SelectionManager:
    """Gestisce la selezione e riproduzione della popolazione"""
    
    def __init__(self, tournament_size: int, mutation_rate: float):
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate

    def tournament_selection(self, 
                           population: List[TradingGene],
                           normalized_scores: np.ndarray) -> Optional[TradingGene]:
        """
        Seleziona genitore usando tournament selection
        
        Args:
            population: Lista di geni
            normalized_scores: Array di scores normalizzati
            
        Returns:
            Gene selezionato o None in caso di errore
        """
        try:
            tournament_idx = np.random.choice(
                len(population),
                size=self.tournament_size,
                replace=False
            )
            
            tournament_scores = normalized_scores[tournament_idx]
            
            if np.all(tournament_scores == tournament_scores[0]) or np.all(np.isnan(tournament_scores)):
                winner_idx = np.random.choice(tournament_idx)
            else:
                winner_idx = tournament_idx[np.argmax(tournament_scores)]
            
            return deepcopy(population[winner_idx])
            
        except Exception as e:
            logger.error(f"Error in tournament selection: {e}")
            return None

    def selection_and_reproduction(self,
                                 population: List[TradingGene],
                                 fitness_scores: List[float],
                                 elite_size: int,
                                 population_size: int,
                                 current_mutation_rate: Optional[float] = None) -> List[TradingGene]:
        """
        Seleziona e riproduce la popolazione
        
        Args:
            population: Lista di geni
            fitness_scores: Lista di fitness scores
            elite_size: Dimensione dell'elite
            population_size: Dimensione della popolazione
            current_mutation_rate: Tasso di mutazione corrente (opzionale)
            
        Returns:
            Nuova popolazione
        """
        try:
            mutation_rate = current_mutation_rate or self.mutation_rate
            
            # Normalizza scores
            fitness_scores_np = np.array(fitness_scores)
            if np.all(np.isnan(fitness_scores_np)):
                normalized_scores = np.ones_like(fitness_scores_np) / len(fitness_scores_np)
            else:
                min_valid = np.nanmin(fitness_scores_np)
                fitness_scores_np = np.nan_to_num(fitness_scores_np, nan=min_valid)
                
                score_range = fitness_scores_np.max() - fitness_scores_np.min()
                if score_range > 0:
                    normalized_scores = (fitness_scores_np - fitness_scores_np.min()) / score_range
                else:
                    normalized_scores = np.ones_like(fitness_scores_np) / len(fitness_scores_np)

            # Seleziona elite
            elite_indices = np.argsort(fitness_scores_np)[-elite_size:]
            new_population: List[TradingGene] = []
            elite_signatures: Set[Tuple] = set()
            
            for idx in elite_indices:
                gene = deepcopy(population[idx])
                signature = self._get_gene_signature(gene)
                if signature not in elite_signatures:
                    elite_signatures.add(signature)
                    new_population.append(gene)

            # Genera nuova popolazione
            attempts = 0
            max_attempts = population_size * 2
            
            while len(new_population) < population_size and attempts < max_attempts:
                try:
                    if np.random.random() < 0.85:  # 85% crossover
                        parent1 = self.tournament_selection(population, normalized_scores)
                        parent2 = self.tournament_selection(population, normalized_scores)
                        if parent1 is not None and parent2 is not None:
                            child = parent1.crossover(parent2)
                            child.mutate(mutation_rate)
                    else:  # 15% mutazione forte
                        parent = self.tournament_selection(population, normalized_scores)
                        if parent is not None:
                            child = deepcopy(parent)
                            child.mutate(mutation_rate * 1.5)
                    
                    if child is not None:
                        child_signature = self._get_gene_signature(child)
                        if child_signature not in elite_signatures:
                            new_population.append(child)
                            elite_signatures.add(child_signature)
                except Exception as e:
                    logger.warning(f"Error in reproduction attempt: {e}")
                
                attempts += 1
            
            # Riempi popolazione se necessario
            while len(new_population) < population_size:
                logger.warning("Adding random individuals to maintain population size")
                new_population.append(TradingGene(random_init=True))
            
            return new_population
            
        except Exception as e:
            logger.error(f"Error in selection and reproduction: {e}")
            return [TradingGene(random_init=True) for _ in range(population_size)]

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
