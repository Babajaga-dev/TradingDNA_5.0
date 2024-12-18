import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)

class AdaptationManager:
    def __init__(self, config):
        # Parametri base
        self.mutation_rate = config.get("genetic.mutation_rate", 0.45)
        self.mutation_decay = config.get("genetic.mutation_decay", 0.995)
        self.diversity_threshold = config.get("genetic.diversity_threshold", 0.25)
        self.restart_threshold = config.get("genetic.restart_threshold", 8)
        self.improvement_threshold = config.get("genetic.improvement_threshold", 0.002)
        self.restart_mutation_multiplier = config.get("genetic.restart_mutation_multiplier", 2.2)
        
        # Parametri adaptive mutation
        adaptive_params = config.get("genetic.adaptive_mutation", {})
        self.plateau_max_factor = adaptive_params.get("plateau_max_factor", 2.0)
        self.plateau_base_factor = adaptive_params.get("plateau_base_factor", 1.0)
        self.improvement_min_factor = adaptive_params.get("improvement_min_factor", 0.5)
        self.improvement_base_factor = adaptive_params.get("improvement_base_factor", 1.0)
        self.fitness_std_multiplier = adaptive_params.get("fitness_std_multiplier", 2.0)
        
        # Validazione parametri
        self._validate_parameters()
        
        logger.info("AdaptationManager inizializzato con parametri:")
        logger.info(f"Plateau factors - Max: {self.plateau_max_factor}, Base: {self.plateau_base_factor}")
        logger.info(f"Improvement factors - Min: {self.improvement_min_factor}, Base: {self.improvement_base_factor}")
        logger.info(f"Fitness std multiplier: {self.fitness_std_multiplier}")

    def _validate_parameters(self):
        """Valida e corregge i parametri se necessario"""
        # Plateau factors
        if self.plateau_max_factor <= self.plateau_base_factor:
            logger.warning("plateau_max_factor <= plateau_base_factor, correzione automatica")
            self.plateau_max_factor = 2.0
            self.plateau_base_factor = 1.0
            
        # Improvement factors
        if self.improvement_min_factor >= self.improvement_base_factor:
            logger.warning("improvement_min_factor >= improvement_base_factor, correzione automatica")
            self.improvement_min_factor = 0.5
            self.improvement_base_factor = 1.0
            
        # Fitness std multiplier
        if self.fitness_std_multiplier <= 0:
            logger.warning("fitness_std_multiplier <= 0, correzione automatica")
            self.fitness_std_multiplier = 2.0

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
            
            # Aumenta mutazione se in plateau usando i fattori configurati
            if plateau_length > 0:
                plateau_factor = self.plateau_base_factor + (
                    (self.plateau_max_factor - self.plateau_base_factor) * 
                    (plateau_length / self.restart_threshold)
                )
                base_rate *= min(self.plateau_max_factor, plateau_factor)
                logger.debug(f"Plateau adjustment: {plateau_factor:.3f}")
            
            # Adatta in base al miglioramento usando i fattori configurati
            if len(generation_stats) > 1:
                last_improvement = (generation_stats[-1]['best_fitness'] - 
                                  generation_stats[-2]['best_fitness'])
                if last_improvement > 0:
                    improvement_factor = self.improvement_base_factor - (
                        (self.improvement_base_factor - self.improvement_min_factor) * 
                        (last_improvement * 2)
                    )
                    base_rate *= max(self.improvement_min_factor, improvement_factor)
                    logger.debug(f"Improvement adjustment: {improvement_factor:.3f}")
            
            # Adatta in base alla deviazione standard del fitness
            if len(generation_stats) > 1:
                fitness_values = [stat['best_fitness'] for stat in generation_stats[-5:]]
                fitness_std = np.std(fitness_values)
                if fitness_std > 0:
                    std_factor = min(2.0, 1.0 + (fitness_std * self.fitness_std_multiplier))
                    base_rate *= std_factor
                    logger.debug(f"Fitness std adjustment: {std_factor:.3f}")
            
            # Applica decay nel tempo
            generation_decay = self.mutation_decay ** generation
            base_rate *= generation_decay
            
            final_rate = min(0.8, max(0.1, base_rate))
            logger.debug(f"Final mutation rate: {final_rate:.3f}")
            return final_rate
            
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
                avg_fitness_std < self.improvement_threshold * self.fitness_std_multiplier and
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
