import logging
import gc
from typing import Optional, Tuple
from datetime import datetime

from ..models.genes.base import TradingGene
from ..models.simulator import TradingSimulator
from ..utils.config import config
from .genetic_stats import OptimizationStats
from .genetic_population import PopulationManager
from .genetic_selection import SelectionManager
from .genetic_device import DeviceManager
from .genetic_monitoring import PerformanceMonitor
from .genetic_evaluation import PopulationEvaluator
from .genetic_adaptation import AdaptationManager

logger = logging.getLogger(__name__)

class ParallelGeneticOptimizer:
    def __init__(self):
        # Inizializza i manager
        self.device_manager = DeviceManager(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.population_manager = PopulationManager(
            population_size=config.get("genetic.population_size", 280),
            mutation_rate=config.get("genetic.mutation_rate", 0.45),
            elite_fraction=config.get("genetic.restart_elite_fraction", 0.12)
        )
        self.selection_manager = SelectionManager(
            tournament_size=config.get("genetic.tournament_size", 5),
            mutation_rate=config.get("genetic.mutation_rate", 0.45)
        )
        self.evaluator = PopulationEvaluator(config, self.device_manager)
        self.adaptation_manager = AdaptationManager(config)
        
        # Parametri base
        self.population_size = config.get("genetic.population_size", 280)
        self.generations = config.get("genetic.generations", 150)
        self.elite_size = config.get("genetic.elite_size", 5)
        
        # Stati
        self.generation_stats = []
        self.best_gene = None
        self.best_fitness = float('-inf')

        logger.info(f"Initialized genetic optimizer with {self.device_manager.num_gpus} devices")
        logger.info(f"Precision: {self.device_manager.precision}")
        logger.info(f"Batch size: {self.evaluator.batch_size}")

    def optimize(self, simulator: TradingSimulator) -> Tuple[Optional[TradingGene], OptimizationStats]:
        """
        Esegue ottimizzazione genetica con supporto CUDA
        
        Args:
            simulator: Simulatore di trading
            
        Returns:
            Tuple con miglior gene e statistiche di ottimizzazione
        """
        try:
            logger.info("Starting enhanced genetic optimization")
            start_time = datetime.now()
            
            self.population_manager.initialize_population()
            
            generations_without_improvement = 0
            best_overall_fitness = float('-inf')
            last_restart_gen = 0
            
            for generation in range(self.generations):
                generation_start = datetime.now()
                logger.info(f"Generation {generation + 1}/{self.generations}")
                
                # Monitora performance se necessario
                if self.performance_monitor.should_check_metrics(generation):
                    self.performance_monitor.check_performance(self.device_manager.devices)
                
                # Valuta popolazione
                fitness_scores, best_gen_fitness, best_gen_gene = self.evaluator.evaluate_population(
                    self.population_manager.population,
                    simulator,
                    self.performance_monitor
                )
                
                # Aggiorna migliori risultati
                if best_gen_fitness > best_overall_fitness and best_gen_gene is not None:
                    best_overall_fitness = best_gen_fitness
                    self.best_gene = best_gen_gene
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Calcola metriche generazione
                current_diversity = self.population_manager.calculate_diversity()
                plateau_length = self.adaptation_manager.calculate_plateau_length(self.generation_stats)
                current_mutation_rate = self.adaptation_manager.calculate_adaptive_mutation_rate(
                    generation, plateau_length, self.generation_stats
                )
                
                # Aggiorna statistiche
                gen_stats = {
                    "generation": generation + 1,
                    "best_fitness": best_gen_fitness,
                    "avg_fitness": sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0,
                    "std_fitness": float(gc.std(fitness_scores)) if fitness_scores else 0.0,
                    "diversity": current_diversity,
                    "mutation_rate": current_mutation_rate,
                    "plateau_length": plateau_length,
                    "elapsed_time": (datetime.now() - generation_start).total_seconds()
                }
                self.generation_stats.append(gen_stats)
                
                # Log progresso
                self.performance_monitor.log_optimization_progress(gen_stats)
                
                # Verifica necessità restart
                if (self.adaptation_manager.check_for_restart(
                    self.generation_stats, current_diversity, last_restart_gen, generation)):
                    logger.info("Performing population restart...")
                    self.population_manager.perform_restart(
                        self.adaptation_manager.restart_mutation_multiplier
                    )
                    last_restart_gen = generation
                    generations_without_improvement = 0
                
                # Verifica necessità injection diversità
                if self.adaptation_manager.should_inject_diversity(
                    generations_without_improvement, current_diversity, generation, self.generations):
                    logger.info("Forced diversity injection")
                    self.population_manager.inject_diversity()
                    generations_without_improvement = 0
                
                # Evolvi popolazione
                if generation < self.generations - 1:
                    self.population_manager.population = self.selection_manager.selection_and_reproduction(
                        population=self.population_manager.population,
                        fitness_scores=fitness_scores,
                        elite_size=self.elite_size,
                        population_size=self.population_size,
                        current_mutation_rate=current_mutation_rate
                    )
                
                # Gestione memoria
                if generation % 10 == 0:
                    if self.device_manager.use_gpu:
                        self.device_manager.manage_cuda_memory()
                    else:
                        gc.collect()
            
            # Calcola statistiche finali
            total_time = (datetime.now() - start_time).total_seconds()
            final_stats = OptimizationStats(
                best_fitness=best_overall_fitness,
                generations=len(self.generation_stats),
                total_time=total_time,
                avg_generation_time=total_time / len(self.generation_stats),
                early_stopped=generations_without_improvement >= self.adaptation_manager.restart_threshold,
                final_population_size=len(self.population_manager.population),
                final_diversity=self.population_manager.calculate_diversity(),
                total_restarts=sum(1 for i in range(1, len(self.generation_stats))
                                if self.generation_stats[i]["diversity"] > 
                                self.generation_stats[i-1]["diversity"] * 1.5)
            )
            
            # Log risultati finali
            logger.info("\nOptimization completed!")
            logger.info(f"Best fitness achieved: {best_overall_fitness:.4f}")
            logger.info(f"Total generations: {final_stats.generations}")
            logger.info(f"Total time: {final_stats.total_time:.2f}s")
            logger.info(f"Number of restarts: {final_stats.total_restarts}")
            
            # Cleanup finale
            if self.device_manager.use_gpu:
                self.device_manager.manage_cuda_memory()
                for device in self.device_manager.devices:
                    try:
                        torch.cuda.synchronize(device)
                    except Exception as e:
                        logger.debug(f"Could not synchronize device {device}: {str(e)}")
            
            return self.best_gene, final_stats
            
        except Exception as e:
            logger.error("Error in genetic optimization:")
            logger.error(str(e))
            return None, OptimizationStats(
                best_fitness=0.0,
                generations=0,
                total_time=0.0,
                avg_generation_time=0.0,
                early_stopped=True,
                final_population_size=0,
                final_diversity=0.0,
                total_restarts=0
            )
