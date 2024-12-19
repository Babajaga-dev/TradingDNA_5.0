import torch
from typing import Dict, Any, Optional

class TradeStats:
    """Calcola e gestisce le statistiche di trading"""
    
    @staticmethod
    def calculate_stats(
        pnl: torch.Tensor,
        equity: torch.Tensor,
        position_active: torch.Tensor,
        initial_capital: float
    ) -> Dict[str, Any]:
        """
        Calcola le statistiche di trading dalla simulazione
        
        Args:
            pnl: Tensore con i P&L
            equity: Tensore con l'equity
            position_active: Tensore booleano delle posizioni attive
            initial_capital: Capitale iniziale
            
        Returns:
            Dizionario con le statistiche calcolate
        """
        # Trova le posizioni chiuse - una posizione è chiusa quando il suo PnL cambia da 0
        closed_positions = pnl != 0
        
        # Filtra solo i trade chiusi e appiattisci il tensore
        trades_pnl = pnl[closed_positions].flatten()
        
        # Statistiche base
        total_trades = len(trades_pnl)
        if total_trades == 0:
            return TradeStats._empty_stats(initial_capital)
            
        winning_trades = torch.sum(trades_pnl > 0).item()
        losing_trades = torch.sum(trades_pnl < 0).item()
        win_rate = (winning_trades / total_trades * 100)
        
        # Calcola average win/loss
        winning_pnl = trades_pnl[trades_pnl > 0]
        losing_pnl = trades_pnl[trades_pnl < 0]
        avg_win = winning_pnl.mean().item() if len(winning_pnl) > 0 else 0
        avg_loss = losing_pnl.mean().item() if len(losing_pnl) > 0 else 0
        
        # Profit factor
        total_profits = winning_pnl.sum().item() if len(winning_pnl) > 0 else 0
        total_losses = abs(losing_pnl.sum().item()) if len(losing_pnl) > 0 else 0
        profit_factor = (total_profits / total_losses) if total_losses != 0 else float('inf')
        
        # Performance metrics
        final_equity = min(equity[-1].item(), initial_capital * 1000)  # Limita a 1000x
        min_equity = max(torch.min(equity).item(), initial_capital * 0.01)  # Limita a -99%
        
        # Calcola il PnL totale sommando il PnL di tutte le posizioni
        total_pl = pnl.sum().item()  # Somma tutti i PnL
        pl_percentage = (total_pl / initial_capital) * 100
        max_drawdown = ((min_equity - initial_capital) / initial_capital) * 100
        
        # Max posizioni contemporanee - somma lungo la dimensione delle posizioni per ogni timestep
        positions_per_timestep = position_active.sum(dim=1)  # Somma le posizioni attive per ogni timestep
        max_concurrent = torch.max(positions_per_timestep).item()  # Prende il massimo numero di posizioni contemporanee
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_pl": total_pl,
            "pl_percentage": pl_percentage,
            "max_drawdown": max_drawdown,
            "max_concurrent_positions": max_concurrent,
            "final_equity": final_equity
        }
        
    @staticmethod
    def _empty_stats(initial_capital: float) -> Dict[str, Any]:
        """
        Restituisce statistiche vuote quando non ci sono trades
        
        Args:
            initial_capital: Capitale iniziale
            
        Returns:
            Dizionario con statistiche vuote
        """
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_pl": 0.0,
            "pl_percentage": 0.0,
            "max_drawdown": 0.0,
            "max_concurrent_positions": 0,
            "final_equity": initial_capital
        }
