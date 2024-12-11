# src/optimization/genetic_utils.py
import torch
import numpy as np
import logging
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)

def to_tensor(data: Union[np.ndarray, torch.Tensor], 
             device: torch.device,
             dtype: torch.dtype) -> torch.Tensor:
    """
    Converte dati in tensor PyTorch
    
    Args:
        data: Dati da convertire
        device: Device su cui allocare il tensor
        dtype: Tipo di dato del tensor
        
    Returns:
        Tensor PyTorch
    """
    try:
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype)
        return torch.tensor(data, dtype=dtype, device=device)
    except Exception as e:
        logger.error(f"Error converting to tensor: {e}")
        return torch.tensor([], dtype=dtype, device=device)

def calculate_fitness(metrics: Dict[str, Any], 
                     min_trades: int,
                     initial_capital: float,
                     weights: Dict[str, Any]) -> float:
    """
    Calcola il fitness score
    
    Args:
        metrics: Dizionario delle metriche
        min_trades: Numero minimo di trade richiesti
        initial_capital: Capitale iniziale
        weights: Pesi per il calcolo del fitness
        
    Returns:
        Score di fitness
    """
    try:
        if metrics["total_trades"] < min_trades:
            return 0.0
            
        profit_weights = weights.get("profit_score", {})
        quality_weights = weights.get("quality_score", {})
        final_weights = weights.get("final_weights", {})
        
        profit_score = (
            profit_weights.get("total_pnl", 0.30) * metrics["total_pnl"] / initial_capital +
            profit_weights.get("max_drawdown", 0.35) * (1 - metrics["max_drawdown"]) +
            profit_weights.get("sharpe_ratio", 0.35) * max(0, metrics["sharpe_ratio"]) / 3
        )
        
        quality_score = (
            quality_weights.get("win_rate", 0.4) * metrics["win_rate"] +
            quality_weights.get("trade_frequency", 0.4) * min(1.0, metrics["total_trades"] / 100)
        )
        
        if "profit_factor" in metrics:
            consistency_score = quality_weights.get("consistency", 0.2) * \
                              (metrics["profit_factor"] - 1) / 2
            quality_score += consistency_score
        
        final_score = (
            final_weights.get("profit", 0.45) * profit_score +
            final_weights.get("quality", 0.45) * quality_score
        )
        
        penalties = 1.0
        if metrics["total_trades"] > 500:
            penalties *= 0.8
        if metrics["max_drawdown"] > 0.3:
            penalties *= 0.7
        if metrics["win_rate"] < 0.4:
            penalties *= 0.9
        
        return max(0.0, final_score * penalties)
        
    except Exception as e:
        logger.error(f"Error calculating fitness: {e}")
        return 0.0
