import torch
import numpy as np
import logging
from typing import Dict, Any
import traceback
from ..utils.config import config

logger = logging.getLogger(__name__)

class MetricsCalculator:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        # Carica parametri da config con conversione esplicita a float
        self.returns_limit = float(config.get("simulator.metrics.returns_limit", 10.0))
        self.min_equity = float(config.get("simulator.metrics.min_equity", 1e-6))

    def calculate_metrics(self, pnl: torch.Tensor, equity: torch.Tensor) -> Dict[str, Any]:
        """
        Calcola metriche con gestione sicura dei valori numerici
        
        Args:
            pnl: Tensor con i profitti/perdite
            equity: Tensor con l'equity curve
            
        Returns:
            Dict con le metriche calcolate
        """
        try:
            # Converti a numpy con controlli di validità
            equity_np = equity.detach().cpu().numpy()
            
            # Applica min_equity
            equity_np = np.where(equity_np < self.min_equity, self.min_equity, equity_np)
            
            if not np.all(np.isfinite(equity_np)):
                logger.warning("Found non-finite values in equity array")
                equity_np = np.nan_to_num(equity_np, nan=self.initial_capital)
            
            # Calcolo trades
            total_trades = int(torch.sum(pnl != 0).item())
            if total_trades == 0:
                return self._get_empty_metrics()

            # Calcolo drawdown
            max_drawdown = self._calculate_drawdown(equity_np)
            
            # Calcolo returns e Sharpe ratio
            sharpe = self._calculate_sharpe_ratio(equity_np)
            
            # Calcolo profit factor e win rate
            profit_metrics = self._calculate_profit_metrics(pnl)
            
            return {
                "total_trades": total_trades,
                "winning_trades": profit_metrics["winning_trades"],
                "win_rate": profit_metrics["win_rate"],
                "total_pnl": float(equity_np[-1] - self.initial_capital),
                "final_capital": float(equity_np[-1]),
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe,
                "profit_factor": profit_metrics["profit_factor"]
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_empty_metrics()

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Restituisce metriche vuote per gestione errori"""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "final_capital": float(self.initial_capital),
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0
        }

    def _calculate_drawdown(self, equity: np.ndarray) -> float:
        """Calcola il drawdown massimo con gestione sicura"""
        try:
            peaks = np.maximum.accumulate(equity)
            min_peak_value = self.min_equity
            peaks = np.where(peaks < min_peak_value, min_peak_value, peaks)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_drawdowns = (peaks - equity) / peaks
                drawdowns = np.where(np.isfinite(raw_drawdowns), raw_drawdowns, 0)
            
            return float(np.nanmax(drawdowns))
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {str(e)}")
            return 0.0

    def _calculate_sharpe_ratio(self, equity: np.ndarray) -> float:
        """Calcola Sharpe ratio con gestione sicura"""
        try:
            equity_shifted = equity[:-1].copy()
            equity_shifted = np.where(equity_shifted < self.min_equity, self.min_equity, equity_shifted)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_returns = np.diff(equity) / equity_shifted
                returns = np.where(np.isfinite(raw_returns), raw_returns, 0)
            
            # Applica returns_limit
            returns = np.clip(returns, -self.returns_limit, self.returns_limit)
            
            if len(returns) > 0:
                returns_std = np.std(returns)
                if returns_std > 0:
                    annualization_factor = float(config.get("simulator.metrics.annualization_factor", 252))
                    return float(np.sqrt(annualization_factor) * (np.mean(returns) / returns_std))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def _calculate_profit_metrics(self, pnl: torch.Tensor) -> Dict[str, Any]:
        """Calcola metriche di profitto con gestione sicura"""
        try:
            # Calcolo winning trades
            winning_trades = int(torch.sum(pnl > 0).item())
            total_trades = int(torch.sum(pnl != 0).item())
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Calcolo profit factor
            positive_pnl = pnl[pnl > 0].sum().item()
            negative_pnl = -pnl[pnl < 0].sum().item()  # Nota il segno negativo
            profit_factor = float(positive_pnl / negative_pnl) if negative_pnl > 0 else 0.0
            
            return {
                "winning_trades": winning_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error calculating profit metrics: {str(e)}")
            return {
                "winning_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0
            }

    def get_performance_metrics(self, metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Restituisce le metriche di performance correnti"""
        if metrics is None:
            return self._get_empty_metrics()
        return metrics
