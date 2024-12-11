from typing import List, Dict, Any
import logging
from dataclasses import dataclass

from ..models.genes.base import TorchGene
from ..models.common import Signal
from ..models.simulator import TradingSimulator
from ..utils.config import config

logger = logging.getLogger(__name__)

@dataclass
class EnsembleMetrics:
    """Metriche per l'ensemble"""
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float

class EnsembleEvaluator:
    """Valuta le performance dell'ensemble"""
    
    def __init__(self, min_trades: int):
        self.min_trades = min_trades
        
        # Carica pesi dal config
        self.profit_weights = config.get("genetic.fitness_weights.profit_score", {})
        self.quality_weights = config.get("genetic.fitness_weights.quality_score", {})
        self.final_weights = config.get("genetic.fitness_weights.final_weights", {})

    def evaluate_ensemble(self, ensemble: List[TorchGene], 
                        simulator: TradingSimulator,
                        signal_combiner) -> float:
        """
        Valuta un ensemble completo
        
        Args:
            ensemble: Lista di geni dell'ensemble
            simulator: Simulatore di trading
            signal_combiner: Combinatore di segnali
            
        Returns:
            Score di fitness
        """
        try:
            # Ottieni segnali da tutti i geni
            gene_signals: List[List[Signal]] = []
            for gene in ensemble:
                signals = gene.generate_signals(simulator.market_state)
                gene_signals.append(signals)
            
            # Combina segnali
            combined_signals = signal_combiner.combine_signals(gene_signals)
            
            # Simula trading con segnali combinati
            simulator._reset_simulation()
            for signal in combined_signals:
                simulator._process_signal(signal)
            
            metrics = simulator.get_performance_metrics()
            
            return self._calculate_fitness(metrics, simulator.initial_capital)
                    
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            return 0.0

    def _calculate_fitness(self, metrics: Dict[str, Any], initial_capital: float) -> float:
        """
        Calcola il fitness score dell'ensemble
        
        Args:
            metrics: Dizionario delle metriche
            initial_capital: Capitale iniziale
            
        Returns:
            Score di fitness
        """
        try:
            if metrics["total_trades"] < self.min_trades:
                return 0.0
            
            profit_score = (
                min(1.0, metrics["total_pnl"] / initial_capital) * 
                self.profit_weights.get("total_pnl", 0.4) +
                (1.0 - metrics["max_drawdown"]) * 
                self.profit_weights.get("max_drawdown", 0.3) +
                max(0.0, metrics["sharpe_ratio"]) * 
                self.profit_weights.get("sharpe_ratio", 0.3)
            )
            
            quality_score = (
                metrics["win_rate"] * 
                self.quality_weights.get("win_rate", 0.6) +
                min(1.0, metrics["total_trades"] / 100.0) * 
                self.quality_weights.get("trade_frequency", 0.4)
            )
            
            return (profit_score * self.final_weights.get("profit", 0.6) + 
                   quality_score * self.final_weights.get("quality", 0.4))
                    
        except Exception as e:
            logger.error(f"Error calculating fitness: {e}")
            return 0.0

    def get_ensemble_metrics(self, metrics: Dict[str, Any]) -> EnsembleMetrics:
        """
        Converte le metriche del simulatore in metriche dell'ensemble
        
        Args:
            metrics: Metriche dal simulatore
            
        Returns:
            Metriche dell'ensemble
        """
        try:
            return EnsembleMetrics(
                total_trades=metrics["total_trades"],
                winning_trades=metrics["winning_trades"],
                win_rate=metrics["win_rate"],
                total_pnl=metrics["total_pnl"],
                max_drawdown=metrics["max_drawdown"],
                sharpe_ratio=metrics["sharpe_ratio"],
                profit_factor=metrics["profit_factor"]
            )
        except Exception as e:
            logger.error(f"Error converting metrics: {e}")
            return EnsembleMetrics(
                total_trades=0,
                winning_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0
            )
