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
        device: Device su cui allocare il tensor (cpu, cuda, xpu)
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
                     weights: Dict[str, Any],
                     ensemble_weights: Dict[str, float]) -> float:
    """
    Calcola il fitness score con supporto per ensemble
    
    Args:
        metrics: Dizionario delle metriche
        min_trades: Numero minimo di trade richiesti
        initial_capital: Capitale iniziale
        weights: Pesi per il calcolo del fitness
        ensemble_weights: Pesi per i diversi tipi di geni nell'ensemble
        
    Returns:
        Score di fitness
    """
    try:
        if metrics["total_trades"] < min_trades:
            return 0.0
            
        profit_weights = weights.get("profit_score", {})
        quality_weights = weights.get("quality_score", {})
        final_weights = weights.get("final_weights", {})
        penalties_config = weights.get("penalties", {})
        
        # Calcolo profit score
        profit_score = (
            profit_weights.get("total_pnl", 0.30) * metrics["total_pnl"] / initial_capital +
            profit_weights.get("max_drawdown", 0.35) * (1 - metrics["max_drawdown"]) +
            profit_weights.get("sharpe_ratio", 0.35) * max(0, metrics["sharpe_ratio"]) / 3
        )
        
        # Calcolo quality score
        quality_score = (
            quality_weights.get("win_rate", 0.4) * metrics["win_rate"] +
            quality_weights.get("trade_frequency", 0.4) * min(1.0, metrics["total_trades"] / 100)
        )
        
        if "profit_factor" in metrics:
            consistency_score = quality_weights.get("consistency", 0.2) * \
                              (metrics["profit_factor"] - 1) / 2
            quality_score += consistency_score
        
        # Calcolo base score
        base_score = (
            final_weights.get("profit", 0.45) * profit_score +
            final_weights.get("quality", 0.45) * quality_score
        )
        
        # Applica pesi ensemble se presenti
        gene_type = metrics.get("gene_type", "base_gene")
        if gene_type in ensemble_weights:
            base_score *= ensemble_weights[gene_type]
        
        # Applica penalità dai parametri di configurazione
        penalties = 1.0
        
        # Penalità max trades
        max_trades_config = penalties_config.get("max_trades", {})
        if metrics["total_trades"] > max_trades_config.get("limit", 500):
            penalties *= max_trades_config.get("penalty", 0.8)
            
        # Penalità max drawdown
        max_drawdown_config = penalties_config.get("max_drawdown", {})
        if metrics["max_drawdown"] > max_drawdown_config.get("limit", 0.3):
            penalties *= max_drawdown_config.get("penalty", 0.7)
            
        # Penalità min win rate
        min_win_rate_config = penalties_config.get("min_win_rate", {})
        if metrics["win_rate"] < min_win_rate_config.get("limit", 0.4):
            penalties *= min_win_rate_config.get("penalty", 0.9)
        
        # Aggiungi diversity bonus se presente
        if "diversity_score" in metrics:
            diversity_bonus = final_weights.get("diversity", 0.1) * metrics["diversity_score"]
            base_score += diversity_bonus
        
        return max(0.0, base_score * penalties)
        
    except Exception as e:
        logger.error(f"Error calculating fitness: {e}")
        return 0.0
