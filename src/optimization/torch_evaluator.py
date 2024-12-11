import torch
import logging
from typing import List, Dict, Any
import traceback

from ..models.genes.base import TradingGene
from ..models.simulator import TradingSimulator

logger = logging.getLogger(__name__)

class TorchEvaluator:
    def __init__(self, simulator: TradingSimulator, data_manager, config):
        self.simulator = simulator
        self.data_manager = data_manager
        self.config = config

    def evaluate_population_parallel(self, 
                                  population: List[TradingGene],
                                  market_data: Dict[str, torch.Tensor]) -> List[float]:
        """
        Valuta popolazione in parallelo su GPU
        
        Args:
            population: Lista di geni da valutare
            market_data: Dati di mercato come tensori
            
        Returns:
            Lista di fitness scores
        """
        try:
            batch_size = self.config.get("genetic.batch_size", 32)
            fitness_scores: List[float] = []

            with self.data_manager.gpu_memory_manager():
                for i in range(0, len(population), batch_size):
                    batch = population[i:i + batch_size]
                    batch_results = self._evaluate_batch(batch, market_data)
                    fitness_scores.extend(batch_results)

            return fitness_scores
            
        except Exception as e:
            logger.error(f"Error evaluating population: {e}")
            return [0.0] * len(population)

    def _evaluate_batch(self, batch: List[TradingGene], 
                      market_data: Dict[str, torch.Tensor]) -> List[float]:
        """
        Valuta un batch di geni
        
        Args:
            batch: Lista di geni da valutare
            market_data: Dati di mercato
            
        Returns:
            Lista di fitness scores per il batch
        """
        try:
            batch_conditions: List[torch.Tensor] = []
            batch_fitness: List[float] = []

            # Genera condizioni di entrata per il batch
            for gene in batch:
                try:
                    conditions = gene.generate_entry_conditions(market_data)
                    batch_conditions.append(conditions)
                except Exception as e:
                    logger.error(f"Error generating conditions for gene: {e}")
                    batch_fitness.append(0.0)
                    continue

            # Converti in tensor e processa su device
            if batch_conditions:
                try:
                    conditions_tensor = torch.stack(batch_conditions).to(self.data_manager.device)
                    
                    # Esegui simulazione vettorizzata
                    with torch.no_grad():
                        metrics = self.simulator.run_simulation_vectorized_torch(
                            conditions_tensor,
                            self.data_manager.device
                        )

                    # Calcola fitness scores
                    for j, gene in enumerate(batch):
                        metrics_dict = {
                            k: v[j].item() if isinstance(v, torch.Tensor) else v
                            for k, v in metrics.items()
                        }
                        fitness = self._calculate_fitness(metrics_dict)
                        batch_fitness.append(fitness)
                        gene.fitness_score = fitness
                        
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    batch_fitness.extend([0.0] * (len(batch) - len(batch_fitness)))

            return batch_fitness
            
        except Exception as e:
            logger.error(f"Error evaluating batch: {e}")
            return [0.0] * len(batch)

    def _calculate_fitness(self, metrics: Dict[str, Any]) -> float:
        """
        Calcola fitness score ottimizzato per GPU
        
        Args:
            metrics: Dizionario delle metriche
            
        Returns:
            Score di fitness
        """
        try:
            if metrics["total_trades"] < self.config.get("genetic.min_trades", 50):
                return 0.0

            weights = self.config.get("genetic.fitness_weights", {})
            
            # Carica parametri di normalizzazione
            sharpe_norm = self.config.get("genetic.optimizer.fitness_calculation.sharpe_normalization", 3.0)
            trade_freq_target = self.config.get("genetic.optimizer.fitness_calculation.trade_frequency_target", 100)
            consistency_norm = self.config.get("genetic.optimizer.fitness_calculation.consistency_normalization", 2.0)
            
            # Calcola componenti del fitness
            profit_score = (
                weights.get("profit_score", {}).get("total_pnl", 0.35) * 
                metrics["total_pnl"] / self.simulator.initial_capital +
                weights.get("profit_score", {}).get("max_drawdown", 0.25) * 
                (1 - metrics["max_drawdown"]) +
                weights.get("profit_score", {}).get("sharpe_ratio", 0.40) * 
                max(0, metrics["sharpe_ratio"]) / sharpe_norm
            )

            quality_score = (
                weights.get("quality_score", {}).get("win_rate", 0.45) * 
                metrics["win_rate"] +
                weights.get("quality_score", {}).get("trade_frequency", 0.25) * 
                min(1.0, metrics["total_trades"] / trade_freq_target)
            )

            if "profit_factor" in metrics:
                consistency = weights.get("quality_score", {}).get("consistency", 0.30) * \
                             (metrics["profit_factor"] - 1) / consistency_norm
                quality_score += consistency

            # Score finale
            final_score = (
                weights.get("final_weights", {}).get("profit", 0.50) * profit_score +
                weights.get("final_weights", {}).get("quality", 0.40) * quality_score
            )

            return max(0.0, final_score)
            
        except Exception as e:
            logger.error(f"Error calculating fitness: {e}")
            return 0.0
