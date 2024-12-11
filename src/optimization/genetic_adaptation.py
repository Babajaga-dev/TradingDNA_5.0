import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)

class AdaptationManager:
    def __init__(self, config):
        # Parametri anti-plateau
        self.mutation_rate = config.get("genetic.mutation_rate", 0.45)
        self.mutation_decay = config.get("genetic.mutation_decay", 0.995)
        self.diversity_threshold = config.get("genetic.diversity_threshold", 0.25)
        self.restart_threshold = config.get("genetic.restart_threshold", 8)
        self.improvement_threshold = config.get("genetic.improvement_threshold", 0.002)
        self.restart_mutation_multiplier = config.get("genetic.restart_mutation_multiplier", 2.2)

    def calculate_plateau_length(self, generation_stats: List[Dict]) -> int:
        """
        Calcola la lunghezza del plateau corrente
        
        Args:
            generation_stats: Lista delle statistiche per generazione
            
        Returns:
            Lunghezza del plateau
        """
        if len(generation_stats) < 2:
            return 0
            
        try:
            current_best = generation_stats[-1]['best_fitness']
            plateau_length = 0
            
            for i in range(len(generation_stats) - 2, -1, -1):
                if abs(generation_stats[i]['best_fitness'] - current_best) < 1e-6:
                    plateau_length += 1
                else:
                    break
                    
            return plateau_length
            
        except Exception as e:
            logger.error(f"Error calculating plateau length: {e}")
            return 0

    def calculate_adaptive_mutation_rate(
        self, 
        generation: int, 
        plateau_length: int,
        generation_stats: List[Dict]
    ) -> float:
        """
        Calcola il tasso di mutazione adattivo
        
        Args:
            generation: Numero della generazione corrente
            plateau_length: Lunghezza del plateau
            generation_stats: Lista delle statistiche per generazione
            
        Returns:
            Tasso di mutazione adattato
        """
        try:
            base_rate = self.mutation_rate
            
            # Aumenta mutazione se in plateau
            if plateau_length > 0:
                plateau_factor = min(2.0, 1.0 + (plateau_length / self.restart_threshold))
                base_rate *= plateau_factor
            
            # Adatta in base al miglioramento
            if len(generation_stats) > 1:
                last_improvement = (generation_stats[-1]['best_fitness'] - 
                                  generation_stats[-2]['best_fitness'])
                if last_improvement > 0:
                    improvement_factor = max(0.5, 1.0 - (last_improvement * 2))
                    base_rate *= improvement_factor
            
            # Applica decay nel tempo
            generation_decay = self.mutation_decay ** generation
            base_rate *= generation_decay
            
            return min(0.8, max(0.1, base_rate))
            
        except Exception as e:
            logger.error(f"Error calculating adaptive mutation rate: {e}")
            return self.mutation_rate

    def check_for_restart(
        self,
        generation_stats: List[Dict],
        current_diversity: float,
        last_restart_gen: int,
        current_gen: int
    ) -> bool:
        """
        Verifica se è necessario un restart della popolazione
        
        Args:
            generation_stats: Lista delle statistiche per generazione
            current_diversity: Diversità corrente della popolazione
            last_restart_gen: Ultima generazione in cui è stato fatto un restart
            current_gen: Generazione corrente
            
        Returns:
            True se è necessario un restart, False altrimenti
        """
        if (len(generation_stats) < self.restart_threshold or
            current_gen - last_restart_gen <= self.restart_threshold):
            return False
            
        try:
            recent_best_fitness = [stat['best_fitness'] 
                                for stat in generation_stats[-self.restart_threshold:]]
            recent_avg_fitness = [stat['avg_fitness']
                               for stat in generation_stats[-self.restart_threshold:]]
            
            best_improvement = max(recent_best_fitness) - min(recent_best_fitness)
            avg_improvement = max(recent_avg_fitness) - min(recent_avg_fitness)
            avg_fitness_std = np.std(recent_avg_fitness)
            
            needs_restart = (
                best_improvement < self.improvement_threshold and
                avg_improvement < self.improvement_threshold and
                avg_fitness_std < self.improvement_threshold * 2 and
                current_diversity < self.diversity_threshold
            )
            
            if needs_restart:
                logger.info(f"Plateau detected for {self.restart_threshold} generations:")
                logger.info(f"Best improvement: {best_improvement:.6f}")
                logger.info(f"Average improvement: {avg_improvement:.6f}")
                logger.info(f"Average fitness std: {avg_fitness_std:.6f}")
                logger.info(f"Current diversity: {current_diversity:.6f}")
            
            return needs_restart
            
        except Exception as e:
            logger.error(f"Error checking for restart: {e}")
            return False

    def should_inject_diversity(
        self,
        generations_without_improvement: int,
        current_diversity: float,
        current_gen: int,
        total_generations: int
    ) -> bool:
        """
        Verifica se è necessario iniettare diversità nella popolazione
        
        Args:
            generations_without_improvement: Generazioni senza miglioramento
            current_diversity: Diversità corrente della popolazione
            current_gen: Generazione corrente
            total_generations: Numero totale di generazioni
            
        Returns:
            True se è necessario iniettare diversità, False altrimenti
        """
        return (generations_without_improvement >= self.restart_threshold and
                current_diversity < self.diversity_threshold and
                current_gen <= total_generations * 0.5)

    def get_restart_mutation_rate(self) -> float:
        """
        Calcola il tasso di mutazione da usare dopo un restart
        
        Returns:
            Tasso di mutazione aumentato
        """
        return min(0.8, self.mutation_rate * self.restart_mutation_multiplier)
