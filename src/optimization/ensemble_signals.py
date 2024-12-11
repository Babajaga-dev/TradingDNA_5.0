from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass

from ..models.common import Signal, SignalType

logger = logging.getLogger(__name__)

class EnsembleSignalCombiner:
    """Combina i segnali dai diversi geni dell'ensemble"""
    
    def __init__(self, ensemble_weights: Dict[str, float]):
        """
        Args:
            ensemble_weights: Dizionario con i pesi per ogni tipo di gene
        """
        self.weights = ensemble_weights

    def combine_signals(self, gene_signals: List[List[Signal]]) -> List[Signal]:
        """
        Combina i segnali usando un sistema di voto pesato
        
        Args:
            gene_signals: Lista di liste di segnali da ogni gene
            
        Returns:
            Lista di segnali combinati
        """
        if not gene_signals:
            return []
        
        try:
            combined_signals = []
            timestamp = gene_signals[0][0].timestamp if gene_signals[0] else None
            
            votes = {
                SignalType.LONG: 0.0,
                SignalType.SHORT: 0.0,
                SignalType.EXIT: 0.0
            }
            
            total_weight = 0.0
            for signals, weight in zip(gene_signals, self.weights.values()):
                if signals:
                    signal_type = signals[0].type
                    votes[signal_type] += weight
                    total_weight += weight
            
            if total_weight == 0:
                return []
                
            for signal_type in votes:
                votes[signal_type] /= total_weight
                
            winning_type = max(votes.items(), key=lambda x: x[1])
            if winning_type[1] >= 0.5:
                stop_losses: List[Tuple[float, float]] = []
                take_profits: List[Tuple[float, float]] = []
                
                for signals, weight in zip(gene_signals, self.weights.values()):
                    if signals and signals[0].type == winning_type[0]:
                        if signals[0].stop_loss is not None:
                            stop_losses.append((signals[0].stop_loss, weight))
                        if signals[0].take_profit is not None:
                            take_profits.append((signals[0].take_profit, weight))
                
                combined_signal = Signal(
                    type=winning_type[0],
                    timestamp=timestamp,
                    price=gene_signals[0][0].price,
                    metadata={"ensemble": True}
                )
                
                if stop_losses:
                    combined_signal.stop_loss = sum(sl * w for sl, w in stop_losses) / sum(w for _, w in stop_losses)
                if take_profits:
                    combined_signal.take_profit = sum(tp * w for tp, w in take_profits) / sum(w for _, w in take_profits)
                
                combined_signals.append(combined_signal)
            
            return combined_signals
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return []
