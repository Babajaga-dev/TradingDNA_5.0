import torch
from typing import Dict, Any

class RiskManager:
    """Gestisce i controlli sul rischio e drawdown"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza il risk manager
        
        Args:
            config: Configurazione con i parametri di rischio
        """
        self.max_drawdown_pct = config.get("trading.risk_management.max_drawdown_pct", 0.15)
        
    def check_max_drawdown(self, current_equity: float, initial_capital: float) -> bool:
        """
        Verifica se è stato raggiunto il max drawdown
        
        Args:
            current_equity: Equity corrente
            initial_capital: Capitale iniziale
            
        Returns:
            True se è stato raggiunto il max drawdown
        """
        drawdown = (initial_capital - current_equity) / initial_capital
        return drawdown >= self.max_drawdown_pct
        
    def validate_position_size(self, position_size: float, current_equity: float) -> bool:
        """
        Verifica se una position size è valida
        
        Args:
            position_size: Size della posizione in percentuale
            current_equity: Equity corrente
            
        Returns:
            True se la position size è valida
        """
        position_value = position_size * current_equity
        return position_value >= 1.0  # Minimo $1 per posizione
        
    def calculate_position_value(self, position_size: float, equity: float) -> float:
        """
        Calcola il valore in dollari di una posizione
        
        Args:
            position_size: Size della posizione in percentuale
            equity: Equity corrente
            
        Returns:
            Valore della posizione in dollari
        """
        return position_size * equity
