# src/optimization/genetic.py
import logging
import gc
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import torch
from datetime import datetime
from contextlib import nullcontext
import traceback

from ..models.genes.base import TradingGene
from ..models.simulator import MarketState, TradingSimulator
from ..utils.config import config
from .genetic_stats import OptimizationStats
from .genetic_utils import to_tensor, calculate_fitness
from .genetic_population import PopulationManager
from .genetic_selection import SelectionManager

logger = logging.getLogger(__name__)

class ParallelGeneticOptimizer:
    def __init__(self):
        # Parametri base
        self.population_size: int = config.get("genetic.population_size", 280)
        self.generations: int = config.get("genetic.generations", 150)
        self.mutation_rate: float = config.get("genetic.mutation_rate", 0.45)
        self.elite_size: int = config.get("genetic.elite_size", 5)
        self.tournament_size: int = config.get("genetic.tournament_size", 5)
        self.min_trades: int = config.get("genetic.min_trades", 50)
        
        # Configurazione CUDA
        self.use_gpu: bool = config.get("genetic.optimizer.use_gpu", False)
        
        # Configura precisione
        self.precision: str = config.get("genetic.optimizer.device_config.precision", "float32")
        self.dtype: torch.dtype = torch.float16 if self.precision == "float16" else torch.float32
        
        try:
            if self.use_gpu and torch.cuda.is_available():
                self.num_gpus: int = torch.cuda.device_count()
                self.devices: List[torch.device] = [torch.device(f"cuda:{i}") for i in range(self.num_gpus)]
                logger.info(f"Using {self.num_gpus} CUDA devices")
                
                # Setup memoria GPU
                self.memory_reserve: int = config.get("genetic.optimizer.device_config.memory_reserve", 2048)
                self.max_batch_size: int = config.get("genetic.optimizer.device_config.max_batch_size", 1024)
                
                # Mixed precision
                self.mixed_precision: bool = config.get("genetic.optimizer.device_config.mixed_precision", False)
                if self.mixed_precision:
                    self.scaler = torch.cuda.amp.GradScaler()
                    
                # CUDA graphs
                self.use_cuda_graphs: bool = config.get("genetic.optimizer.device_config.cuda_graph", False)
                
                # Distribuzione popolazione
                self.genes_per_gpu: int = self.population_size // self.num_gpus
                self.batch_size: int = min(self.max_batch_size, 
                                         config.get("genetic.batch_size", 64))
                
                # Log GPU info
                logger.info(f"CUDA enabled: {torch.cuda.is_available()}")
                logger.info(f"Mixed precision: {self.mixed_precision}")
                for i in range(self.num_gpus):
                    try:
                        device_name = torch.cuda.get_device_name(i)
                        memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                        logger.info(f"GPU {i}: {device_name}")
                        logger.info(f"Memory allocated: {memory_allocated:.2f} GB")
                    except Exception as e:
                        logger.warning(f"Could not get info for GPU {i}: {str(e)}")
            else:
                if self.use_gpu:
                    logger.warning("GPU requested but not available. Falling back to CPU.")
                self.devices = [torch.device("cpu")]
                self.num_gpus = 1
                self.genes_per_gpu = self.population_size
                self.batch_size = config.get("genetic.batch_size", 32)
                self.mixed_precision = False
                self.use_cuda_graphs = False
                
                # Setup CPU threads
                cpu_threads = config.get("genetic.optimizer.torch_threads", 4)
                torch.set_num_threads(cpu_threads)
                logger.info(f"Using CPU with {cpu_threads} threads")
        except Exception as e:
            logger.error(f"Error during device setup: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback to CPU
            self.devices = [torch.device("cpu")]
            self.num_gpus = 1
            self.genes_per_gpu = self.population_size
            self.batch_size = config.get("genetic.batch_size", 32)
            self.mixed_precision = False
            self.use_cuda_graphs = False
            cpu_threads = config.get("genetic.optimizer.torch_threads", 4)
            torch.set_num_threads(cpu_threads)
            logger.info("Fallback to CPU mode due to error")

        # Parametri anti-plateau
        self.mutation_decay: float = config.get("genetic.mutation_decay", 0.995)
        self.diversity_threshold: float = config.get("genetic.diversity_threshold", 0.25)
        self.restart_threshold: int = config.get("genetic.restart_threshold", 8)
        self.improvement_threshold: float = config.get("genetic.improvement_threshold", 0.002)
        self.restart_elite_fraction: float = config.get("genetic.restart_elite_fraction", 0.12)
        self.restart_mutation_multiplier: float = config.get("genetic.restart_mutation_multiplier", 2.2)
        
        # Managers
        self.population_manager = PopulationManager(
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
            elite_fraction=self.restart_elite_fraction
        )
        self.selection_manager = SelectionManager(
            tournament_size=self.tournament_size,
            mutation_rate=self.mutation_rate
        )
        
        # Stati
        self.generation_stats: List[Dict[str, Any]] = []
        self.best_gene: Optional[TradingGene] = None
        self.best_fitness: float = float('-inf')
        self.precalculated_data: Optional[Dict[str, np.ndarray]] = None

        logger.info(f"Initialized genetic optimizer with {self.num_gpus} devices")
        logger.info(f"Precision: {self.precision}")
        logger.info(f"Batch size: {self.batch_size}")

    def _evaluate_population(self, simulator: TradingSimulator) -> Tuple[List[float], float, Optional[TradingGene]]:
        """
        Valuta popolazione con supporto multi-GPU
        
        Args:
            simulator: Simulatore di trading
            
        Returns:
            Tuple con lista di fitness scores, miglior fitness e miglior gene
        """
        try:
            fitness_scores: List[float] = []
            best_generation_fitness = float('-inf')
            best_generation_gene = None
            
            # Converti gli indicatori precalcolati in numpy arrays
            if self.precalculated_data is None:
                self.precalculated_data = {
                    k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in simulator.indicators_cache.items()
                }
                # Aggiungi CLOSE come indicatore
                self.precalculated_data["CLOSE"] = simulator.market_state.close
            
            chunks = np.array_split(self.population_manager.population, self.num_gpus)
            
            for gpu_id, chunk in enumerate(chunks):
                device = self.devices[gpu_id % self.num_gpus]
                
                for i in range(0, len(chunk), self.batch_size):
                    batch = chunk[i:i + self.batch_size]
                    batch_fitness: List[float] = []
                    
                    if self.use_gpu and self.use_cuda_graphs:
                        static_graph = None
                    
                    for gene in batch:
                        try:
                            entry_conditions = gene.generate_entry_conditions(self.precalculated_data)
                            
                            if self.use_gpu:
                                entry_conditions = to_tensor(entry_conditions, device, self.dtype)
                            
                            if self.use_gpu and self.use_cuda_graphs:
                                if static_graph is None:
                                    with torch.cuda.graph(static_graph):
                                        metrics = simulator.run_simulation_vectorized(entry_conditions)
                                else:
                                    metrics = simulator.run_simulation_vectorized(entry_conditions)
                            else:
                                metrics = simulator.run_simulation_vectorized(entry_conditions)
                            
                            fitness = calculate_fitness(
                                metrics=metrics,
                                min_trades=self.min_trades,
                                initial_capital=simulator.initial_capital,
                                weights=config.get("genetic.fitness_weights", {})
                            )
                            gene.fitness_score = fitness
                            batch_fitness.append(fitness)
                            
                            if fitness > best_generation_fitness:
                                best_generation_fitness = fitness
                                best_generation_gene = gene
                                
                        except Exception as e:
                            logger.error(f"Error evaluating gene: {str(e)}")
                            batch_fitness.append(0.0)
                    
                    fitness_scores.extend(batch_fitness)
                    
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                        if gpu_id == 0:
                            try:
                                memory_allocated = torch.cuda.memory_allocated(device) / 1e9
                                logger.debug(f"GPU memory used: {memory_allocated:.2f} GB")
                            except Exception as e:
                                logger.debug(f"Could not get GPU memory info: {str(e)}")
            
            return fitness_scores, best_generation_fitness, best_generation_gene
            
        except Exception as e:
            logger.error(f"Error in population evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            return [], float('-inf'), None

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
                
                fitness_scores, best_gen_fitness, best_gen_gene = self._evaluate_population(simulator)
                
                if best_gen_fitness > best_overall_fitness and best_gen_gene is not None:
                    best_overall_fitness = best_gen_fitness
                    self.best_gene = best_gen_gene
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                current_diversity = self.population_manager.calculate_diversity()
                plateau_length = self._calculate_plateau_length()
                current_mutation_rate = self._adaptive_mutation_rate(generation, plateau_length)
                
                gen_stats = {
                    "generation": generation + 1,
                    "best_fitness": best_gen_fitness,
                    "avg_fitness": np.mean(fitness_scores) if fitness_scores else 0.0,
                    "std_fitness": np.std(fitness_scores) if fitness_scores else 0.0,
                    "diversity": current_diversity,
                    "mutation_rate": current_mutation_rate,
                    "plateau_length": plateau_length,
                    "elapsed_time": (datetime.now() - generation_start).total_seconds()
                }
                self.generation_stats.append(gen_stats)
                
                logger.info(f"Best Fitness: {gen_stats['best_fitness']:.4f}")
                logger.info(f"Average Fitness: {gen_stats['avg_fitness']:.4f}")
                logger.info(f"Population Diversity: {gen_stats['diversity']:.4f}")
                logger.info(f"Current Mutation Rate: {gen_stats['mutation_rate']:.4f}")
                logger.info(f"Time: {gen_stats['elapsed_time']:.2f}s")
                
                if (self._check_for_restart() and 
                    generation - last_restart_gen > self.restart_threshold):
                    logger.info("Performing population restart...")
                    self.population_manager.perform_restart(self.restart_mutation_multiplier)
                    last_restart_gen = generation
                    generations_without_improvement = 0
                
                if (generations_without_improvement >= self.restart_threshold and 
                    current_diversity < self.diversity_threshold):
                    if generation > self.generations * 0.5:
                        logger.info("Early stopping - No improvement and low diversity")
                        break
                    else:
                        logger.info("Forced diversity injection")
                        self.population_manager.inject_diversity()
                        generations_without_improvement = 0
                
                if generation < self.generations - 1:
                    self.population_manager.population = self.selection_manager.selection_and_reproduction(
                        population=self.population_manager.population,
                        fitness_scores=fitness_scores,
                        elite_size=self.elite_size,
                        population_size=self.population_size,
                        current_mutation_rate=current_mutation_rate
                    )
                
                if generation % 10 == 0:
                    gc.collect()
                    if self.use_gpu:
                        torch.cuda.empty_cache()
            
            total_time = (datetime.now() - start_time).total_seconds()
            final_stats = OptimizationStats(
                best_fitness=best_overall_fitness,
                generations=len(self.generation_stats),
                total_time=total_time,
                avg_generation_time=total_time / len(self.generation_stats),
                early_stopped=generations_without_improvement >= self.restart_threshold,
                final_population_size=len(self.population_manager.population),
                final_diversity=self.population_manager.calculate_diversity(),
                total_restarts=sum(1 for i in range(1, len(self.generation_stats))
                                if self.generation_stats[i]["diversity"] > 
                                self.generation_stats[i-1]["diversity"] * 1.5)
            )
            
            logger.info("\nOptimization completed!")
            logger.info(f"Best fitness achieved: {best_overall_fitness:.4f}")
            logger.info(f"Total generations: {final_stats.generations}")
            logger.info(f"Total time: {final_stats.total_time:.2f}s")
            logger.info(f"Number of restarts: {final_stats.total_restarts}")
            
            if self.use_gpu:
                torch.cuda.empty_cache()
                for device in self.devices:
                    try:
                        torch.cuda.synchronize(device)
                    except Exception as e:
                        logger.debug(f"Could not synchronize device {device}: {str(e)}")
            
            return self.best_gene, final_stats
            
        except Exception as e:
            logger.error("Error in genetic optimization:")
            logger.error(str(e))
            logger.error(traceback.format_exc())
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

    def _calculate_plateau_length(self) -> int:
        """
        Calcola la lunghezza del plateau
        
        Returns:
            Lunghezza del plateau
        """
        if len(self.generation_stats) < 2:
            return 0
            
        try:
            current_best = self.generation_stats[-1]['best_fitness']
            plateau_length = 0
            
            for i in range(len(self.generation_stats) - 2, -1, -1):
                if abs(self.generation_stats[i]['best_fitness'] - current_best) < 1e-6:
                    plateau_length += 1
                else:
                    break
                    
            return plateau_length
            
        except Exception as e:
            logger.error(f"Error calculating plateau length: {e}")
            return 0

    def _adaptive_mutation_rate(self, generation: int, plateau_length: int) -> float:
        """
        Calcola il tasso di mutazione adattivo
        
        Args:
            generation: Numero della generazione corrente
            plateau_length: Lunghezza del plateau
            
        Returns:
            Tasso di mutazione adattato
        """
        try:
            base_rate = self.mutation_rate
            
            if plateau_length > 0:
                plateau_factor = min(2.0, 1.0 + (plateau_length / self.restart_threshold))
                base_rate *= plateau_factor
            
            if len(self.generation_stats) > 1:
                last_improvement = (self.generation_stats[-1]['best_fitness'] - 
                                  self.generation_stats[-2]['best_fitness'])
                if last_improvement > 0:
                    improvement_factor = max(0.5, 1.0 - (last_improvement * 2))
                    base_rate *= improvement_factor
            
            return min(0.8, max(0.1, base_rate))
            
        except Exception as e:
            logger.error(f"Error calculating adaptive mutation rate: {e}")
            return self.mutation_rate

    def _check_for_restart(self) -> bool:
        """
        Verifica se è necessario un restart
        
        Returns:
            True se è necessario un restart, False altrimenti
        """
        if len(self.generation_stats) < self.restart_threshold:
            return False
            
        try:
            recent_best_fitness = [stat['best_fitness'] 
                                for stat in self.generation_stats[-self.restart_threshold:]]
            recent_avg_fitness = [stat['avg_fitness']
                               for stat in self.generation_stats[-self.restart_threshold:]]
            
            best_improvement = max(recent_best_fitness) - min(recent_best_fitness)
            avg_improvement = max(recent_avg_fitness) - min(recent_avg_fitness)
            avg_fitness_std = np.std(recent_avg_fitness)
            
            needs_restart = (
                best_improvement < self.improvement_threshold and
                avg_improvement < self.improvement_threshold and
                avg_fitness_std < self.improvement_threshold * 2
            )
            
            if needs_restart:
                logger.info(f"Plateau detected for {self.restart_threshold} generations:")
                logger.info(f"Best improvement: {best_improvement:.6f}")
                logger.info(f"Average improvement: {avg_improvement:.6f}")
                logger.info(f"Average fitness std: {avg_fitness_std:.6f}")
            
            return needs_restart
            
        except Exception as e:
            logger.error(f"Error checking for restart: {e}")
            return False
