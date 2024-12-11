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
        """
        Inizializza il selection manager.
        
        Args:
            tournament_size: Dimensione del torneo per la selezione
            mutation_rate: Tasso base di mutazione
        """
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate

    def _get_gene_signature(self, gene: TradingGene) -> Tuple:
        """
        Genera una signature unica per il gene.
        
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

    def tournament_selection(self, 
                           population: List[TradingGene],
                           normalized_scores: np.ndarray) -> Optional[TradingGene]:
        """
        Seleziona un gene usando tournament selection.
        
        Args:
            population: Lista di geni
            normalized_scores: Array di scores normalizzati
            
        Returns:
            Gene selezionato o None in caso di errore
        """
        try:
            pop_size = len(population)
            if pop_size == 0:
                logger.error("Empty population in tournament selection")
                return None

            # Verifica che gli array siano della stessa dimensione
            if pop_size != len(normalized_scores):
                logger.error(f"Population size ({pop_size}) and scores size ({len(normalized_scores)}) mismatch")
                return None
                
            # Assicurati che tournament_size non sia più grande della popolazione
            actual_tournament_size = min(self.tournament_size, pop_size)
            if actual_tournament_size < 1:
                logger.error(f"Invalid tournament size: {actual_tournament_size}")
                return None

            # Crea indici validi per il torneo
            valid_indices = np.arange(pop_size)
            tournament_idx = np.random.choice(valid_indices, size=actual_tournament_size, replace=False)
            
            tournament_scores = normalized_scores[tournament_idx]
            
            # Seleziona il vincitore
            if np.all(np.isnan(tournament_scores)) or np.all(tournament_scores == tournament_scores[0]):
                winner_idx = tournament_idx[np.random.randint(actual_tournament_size)]
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
        Seleziona e riproduce la popolazione.
        
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
            if not population or len(population) == 0:
                logger.error("Empty population in selection and reproduction")
                return [TradingGene(random_init=True) for _ in range(population_size)]

            # Verifica lunghezza fitness scores
            if len(population) != len(fitness_scores):
                logger.error(f"Population size ({len(population)}) and fitness scores size ({len(fitness_scores)}) mismatch")
                return [TradingGene(random_init=True) for _ in range(population_size)]

            mutation_rate = current_mutation_rate or self.mutation_rate
            
            # Normalizza scores e gestisci NaN
            fitness_scores_np = np.array(fitness_scores)
            min_valid = np.nanmin(fitness_scores_np) if not np.all(np.isnan(fitness_scores_np)) else 0.0
            fitness_scores_np = np.nan_to_num(fitness_scores_np, nan=min_valid)
            
            score_range = np.ptp(fitness_scores_np)
            if score_range > 0:
                normalized_scores = (fitness_scores_np - np.min(fitness_scores_np)) / score_range
            else:
                normalized_scores = np.ones_like(fitness_scores_np) / len(fitness_scores_np)

            # Elite selection
            elite_size = min(elite_size, len(population))
            elite_indices = np.argsort(fitness_scores_np)[-elite_size:]
            new_population = []
            elite_signatures: Set[Tuple] = set()

            # Preserva elite
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
                    # Inizializza child esplicitamente come None
                    child = None
                    
                    # Crossover (85% probabilità)
                    if np.random.random() < 0.85:
                        parent1 = self.tournament_selection(population, normalized_scores)
                        parent2 = self.tournament_selection(population, normalized_scores)
                        
                        if parent1 is not None and parent2 is not None:
                            child = parent1.crossover(parent2)
                            if child is not None:
                                child.mutate(mutation_rate)
                    
                    # Strong mutation (15% probabilità)
                    else:
                        parent = self.tournament_selection(population, normalized_scores)
                        if parent is not None:
                            child = deepcopy(parent)
                            if child is not None:
                                child.mutate(mutation_rate * 1.5)

                    # Aggiungi child se valido e unico
                    if child is not None:
                        child_signature = self._get_gene_signature(child)
                        if child_signature not in elite_signatures:
                            new_population.append(child)
                            elite_signatures.add(child_signature)
                            
                except Exception as e:
                    logger.warning(f"Error in reproduction attempt: {str(e)}")
                
                attempts += 1

            # Riempi eventuali spazi vuoti con individui random
            while len(new_population) < population_size:
                logger.warning(f"Adding random individual ({len(new_population)}/{population_size})")
                new_gene = TradingGene(random_init=True)
                new_population.append(new_gene)

            # Verifica finale
            if len(new_population) != population_size:
                logger.error(f"Population size mismatch after reproduction. Expected: {population_size}, Got: {len(new_population)}")
                # Aggiusta la dimensione se necessario
                if len(new_population) > population_size:
                    new_population = new_population[:population_size]
                while len(new_population) < population_size:
                    new_population.append(TradingGene(random_init=True))

            return new_population

        except Exception as e:
            logger.error(f"Error in selection and reproduction: {str(e)}")
            logger.exception("Traceback:")
            return [TradingGene(random_init=True) for _ in range(population_size)]