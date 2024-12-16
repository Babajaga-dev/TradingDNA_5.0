import logging
import numpy as np
import torch
import intel_extension_for_pytorch as ipex
from typing import List, Tuple, Dict, Optional
from contextlib import nullcontext

from ..models.genes.base import TradingGene
from ..models.simulator import TradingSimulator
from .genetic_utils import to_tensor, calculate_fitness

logger = logging.getLogger(__name__)

class PopulationEvaluator:
    def __init__(self, config, device_manager):
        self.min_trades = config.get("genetic.min_trades", 50)
        self.batch_size = config.get("genetic.batch_size", 64)
        self.device_manager = device_manager
        self.fitness_weights = config.get("genetic.fitness_weights", {
            "profit_score": {
                "total_pnl": 0.35,
                "max_drawdown": 0.25,
                "sharpe_ratio": 0.40
            },
            "quality_score": {
                "win_rate": 0.45,
                "trade_frequency": 0.25,
                "consistency": 0.30
            },
            "final_weights": {
                "profit": 0.50,
                "quality": 0.40,
                "diversity": 0.10
            }
        })
        self.ensemble_weights = config.get("genetic.ensemble_weights", {
            "base_gene": 0.3,
            "momentum_gene": 0.25,
            "volatility_gene": 0.25,
            "pattern_gene": 0.2
        })
        self.precalculated_data = None

    def evaluate_population(
        self, 
        population: List[TradingGene], 
        simulator: TradingSimulator,
        performance_monitor
    ) -> Tuple[List[float], float, Optional[TradingGene]]:
        """
        Valuta l'intera popolazione
        
        Args:
            population: Lista di geni da valutare
            simulator: Simulatore di trading
            performance_monitor: Monitor delle performance
            
        Returns:
            Tuple con lista dei punteggi fitness, miglior fitness e miglior gene
        """
        try:
            fitness_scores: List[float] = []
            best_generation_fitness = float('-inf')
            best_generation_gene = None
            
            if self.precalculated_data is None:
                self._prepare_precalculated_data(simulator)
            
            chunks = np.array_split(population, self.device_manager.num_gpus)
            
            for gpu_id, chunk in enumerate(chunks):
                device = self.device_manager.devices[gpu_id % self.device_manager.num_gpus]
                chunk_results = self._evaluate_chunk(chunk, device, simulator, performance_monitor)
                
                fitness_scores.extend(chunk_results[0])
                if chunk_results[1] > best_generation_fitness:
                    best_generation_fitness = chunk_results[1]
                    best_generation_gene = chunk_results[2]
                    
                if self.device_manager.use_gpu and gpu_id == 0:
                    try:
                        if self.device_manager.gpu_backend == "arc":
                            memory_allocated = torch.xpu.memory_allocated() / 1e9
                            logger.debug(f"XPU memory used: {memory_allocated:.2f} GB")
                        else:
                            memory_allocated = torch.cuda.memory_allocated(device) / 1e9
                            logger.debug(f"GPU memory used: {memory_allocated:.2f} GB")
                    except Exception as e:
                        logger.debug(f"Could not get GPU memory info: {str(e)}")
            
            return fitness_scores, best_generation_fitness, best_generation_gene
            
        except Exception as e:
            logger.error(f"Error in population evaluation: {str(e)}")
            return [], float('-inf'), None

    def _prepare_precalculated_data(self, simulator: TradingSimulator) -> None:
        """Prepara i dati precalcolati per la valutazione"""
        self.precalculated_data = {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in simulator.indicators_cache.items()
        }
        self.precalculated_data["CLOSE"] = simulator.market_state.close

    def _evaluate_chunk(
        self, 
        chunk: List[TradingGene],
        device: torch.device,
        simulator: TradingSimulator,
        performance_monitor
    ) -> Tuple[List[float], float, Optional[TradingGene]]:
        """
        Valuta un chunk della popolazione su un dispositivo specifico
        
        Args:
            chunk: Lista di geni da valutare
            device: Dispositivo su cui eseguire la valutazione
            simulator: Simulatore di trading
            performance_monitor: Monitor delle performance
            
        Returns:
            Tuple con lista dei punteggi fitness, miglior fitness e miglior gene
        """
        chunk_fitness: List[float] = []
        best_chunk_fitness = float('-inf')
        best_chunk_gene = None
        
        for i in range(0, len(chunk), self.batch_size):
            if performance_monitor.should_check_metrics(i):
                performance_monitor.check_performance([device])
                
            batch = chunk[i:i + self.batch_size]
            batch_results = self._evaluate_batch(batch, device, simulator)
            
            chunk_fitness.extend(batch_results[0])
            if batch_results[1] > best_chunk_fitness:
                best_chunk_fitness = batch_results[1]
                best_chunk_gene = batch_results[2]
        
        return chunk_fitness, best_chunk_fitness, best_chunk_gene

    def _evaluate_batch(
        self,
        batch: List[TradingGene],
        device: torch.device,
        simulator: TradingSimulator
    ) -> Tuple[List[float], float, Optional[TradingGene]]:
        """
        Valuta un batch di geni
        
        Args:
            batch: Lista di geni da valutare
            device: Dispositivo su cui eseguire la valutazione
            simulator: Simulatore di trading
            
        Returns:
            Tuple con lista dei punteggi fitness, miglior fitness e miglior gene
        """
        batch_fitness: List[float] = []
        best_batch_fitness = float('-inf')
        best_batch_gene = None
        
        for gene in batch:
            try:
                entry_conditions = gene.generate_entry_conditions(self.precalculated_data)
                
                if self.device_manager.use_gpu:
                    entry_conditions = to_tensor(entry_conditions, device, self.device_manager.dtype)
                
                # Configura mixed precision in base al backend
                if self.device_manager.mixed_precision:
                    if self.device_manager.gpu_backend == "arc":
                        # XPU mixed precision è già configurata attraverso ipex.optimize()
                        autocast_ctx = nullcontext()
                    elif self.device_manager.gpu_backend == "cuda":
                        autocast_ctx = torch.amp.autocast('cuda')
                    else:
                        autocast_ctx = nullcontext()
                else:
                    autocast_ctx = nullcontext()
                
                with autocast_ctx:
                    metrics = simulator.run_simulation_vectorized(entry_conditions, gene)
                
                fitness = calculate_fitness(
                    metrics=metrics,
                    min_trades=self.min_trades,
                    initial_capital=simulator.initial_capital,
                    weights=self.fitness_weights,
                    ensemble_weights=self.ensemble_weights
                )
                gene.fitness_score = fitness
                batch_fitness.append(fitness)
                
                if fitness > best_batch_fitness:
                    best_batch_fitness = fitness
                    best_batch_gene = gene
                    
            except Exception as e:
                logger.error(f"Error evaluating gene: {str(e)}")
                batch_fitness.append(0.0)
        
        return batch_fitness, best_batch_fitness, best_batch_gene
