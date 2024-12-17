import torch
import numpy as np
import logging
from typing import Dict, Any
import traceback
from ..utils.config import config

logger = logging.getLogger(__name__)


class MetricsCalculator:
    def __init__(self, initial_capital: float):
        self.initial_capital = float(initial_capital)
        self.min_equity = 1e-3
        self.max_equity_multiplier = 5.0
        self.compute_dtype = torch.float32

    def calculate_metrics(self, pnl: torch.Tensor, equity: torch.Tensor) -> Dict[str, Any]:
        """Calcola le metriche con protezione robusta dai valori non finiti"""
        try:
            # Converti a float32 per calcoli precisi
            equity = equity.to(dtype=self.compute_dtype)
            pnl = pnl.to(dtype=self.compute_dtype)

            # Assicura che l'equity non scenda sotto il minimo
            equity = torch.maximum(equity, 
                torch.tensor(self.min_equity, dtype=self.compute_dtype, device=equity.device))

            # Limita l'equity massima
            max_equity = self.initial_capital * self.max_equity_multiplier
            equity = torch.minimum(equity, 
                torch.tensor(max_equity, dtype=self.compute_dtype, device=equity.device))

            # Calcola metriche di base
            total_trades = int(torch.sum(pnl != 0).item())
            if total_trades == 0:
                return self._get_empty_metrics()

            # Calcola PnL totale e finale
            total_pnl = float(equity[-1].item() - self.initial_capital)
            final_capital = float(equity[-1].item())

            # Calcola metriche di trade
            winning_trades = int(torch.sum(pnl > 0).item())
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            # Calcola drawdown in modo sicuro
            peaks, _ = torch.cummax(equity, dim=0)  # Usa cummax invece di maximum.accumulate
            drawdowns = (peaks - equity) / peaks
            drawdowns = torch.nan_to_num(drawdowns, nan=0.0)
            max_drawdown = float(torch.max(drawdowns).item())

            # Calcola Sharpe ratio in modo sicuro
            returns = self._calculate_returns(equity)
            sharpe_ratio = self._calculate_sharpe(returns)

            # Calcola profit factor
            profit_factor = self._calculate_profit_factor(pnl)

            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "final_capital": final_capital,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "profit_factor": profit_factor
            }

        except Exception as e:
            logger.error(f"Errore nel calcolo delle metriche: {str(e)}")
            return self._get_empty_metrics()

    def _calculate_returns(self, equity: torch.Tensor) -> torch.Tensor:
        """Calcola i returns in modo sicuro"""
        try:
            # Usa una finestra mobile per calcolare i returns
            prev_equity = torch.roll(equity, shifts=1)
            prev_equity[0] = self.initial_capital

            # Usa clipping per limitare i returns estremi
            returns = (equity - prev_equity) / torch.maximum(
                prev_equity,
                torch.tensor(self.min_equity, dtype=self.compute_dtype, device=equity.device)
            )

            # Limita i returns a valori ragionevoli (-50% a +50%)
            returns = torch.clamp(returns, min=-0.5, max=0.5)

            # Sostituisci eventuali NaN
            returns = torch.nan_to_num(returns, nan=0.0)

            return returns[1:]  # Ignora il primo valore che non è un vero return

        except Exception as e:
            logger.error(f"Errore nel calcolo dei returns: {str(e)}")
            return torch.zeros_like(equity[1:])

    def _calculate_sharpe(self, returns: torch.Tensor) -> float:
        """Calcola Sharpe ratio in modo sicuro"""
        try:
            if len(returns) < 2:
                return 0.0

            # Rimuovi outliers
            mean = torch.mean(returns)
            std = torch.std(returns)
            if std == 0:
                return 0.0

            z_scores = (returns - mean) / std
            valid_returns = returns[torch.abs(z_scores) <= 3]  # keep within 3 std

            if len(valid_returns) < 2:
                return 0.0

            # Ricalcola statistiche senza outliers
            mean = torch.mean(valid_returns)
            std = torch.std(valid_returns)
            
            if std == 0:
                return 0.0

            annualization = 252  # Annualizzazione per dati giornalieri
            return float(torch.sqrt(torch.tensor(annualization)) * mean / std)

        except Exception as e:
            logger.error(f"Errore nel calcolo dello Sharpe ratio: {str(e)}")
            return 0.0

    def _calculate_profit_factor(self, pnl: torch.Tensor) -> float:
        """Calcola profit factor in modo sicuro"""
        try:
            gains = torch.sum(torch.where(pnl > 0, pnl, torch.zeros_like(pnl)))
            losses = torch.abs(torch.sum(torch.where(pnl < 0, pnl, torch.zeros_like(pnl))))

            if losses == 0:
                return 1.0 if gains > 0 else 0.0

            return float(gains / losses)

        except Exception as e:
            logger.error(f"Errore nel calcolo del profit factor: {str(e)}")
            return 0.0

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Restituisce metriche vuote"""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "final_capital": self.initial_capital,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0
        }
