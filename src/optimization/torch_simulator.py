import torch
import logging
from typing import Dict, Any
import traceback

logger = logging.getLogger(__name__)

class TorchSimulator:
    def __init__(self, config):
        self.config = config

    def run_simulation_vectorized(self, 
                                entry_conditions: torch.Tensor,
                                prices: torch.Tensor,
                                device: torch.device,
                                initial_capital: float) -> Dict[str, torch.Tensor]:
        """
        Versione PyTorch della simulazione vettorizzata
        
        Args:
            entry_conditions: Condizioni di entrata
            prices: Prezzi di mercato
            device: Device PyTorch
            initial_capital: Capitale iniziale
            
        Returns:
            Dizionario delle metriche
        """
        try:
            # Converti parametri in tensori
            position_size_pct = self.config.get("trading.position.size_pct", 5) / 100
            stop_loss_pct = self.config.get("trading.position.stop_loss_pct", 2) / 100
            take_profit_pct = self.config.get("trading.position.take_profit_pct", 4) / 100

            # Prepara tensori su device
            position_active = torch.zeros_like(prices, dtype=torch.bool, device=device)
            entry_prices = torch.zeros_like(prices, device=device)
            pnl = torch.zeros_like(prices, device=device)
            equity = torch.ones_like(prices, device=device) * initial_capital

            # Simulazione
            for i in range(1, len(prices)):
                self._process_timestep(
                    i, prices, entry_conditions,
                    position_active, entry_prices, pnl, equity,
                    position_size_pct, stop_loss_pct, take_profit_pct
                )

            # Calcola metriche
            metrics = self._calculate_metrics(pnl, equity, initial_capital)
            return metrics
            
        except Exception as e:
            logger.error(f"Error in vectorized simulation: {e}")
            return self._get_empty_metrics(device, initial_capital)

    def _process_timestep(self, i: int,
                        prices: torch.Tensor,
                        entry_conditions: torch.Tensor,
                        position_active: torch.Tensor,
                        entry_prices: torch.Tensor,
                        pnl: torch.Tensor,
                        equity: torch.Tensor,
                        position_size_pct: float,
                        stop_loss_pct: float,
                        take_profit_pct: float) -> None:
        """Processa un singolo timestep della simulazione"""
        current_price = prices[i]
        
        # Entry
        mask_entry = entry_conditions[i] & ~position_active[i-1]
        position_active[i:][mask_entry] = True
        entry_prices[i:][mask_entry] = current_price
        
        # Check exit
        mask_active = position_active[i-1]
        if torch.any(mask_active):
            entry_price = entry_prices[i-1][mask_active]
            price_change = (current_price - entry_price) / entry_price
            
            exit_mask = (
                (price_change <= -stop_loss_pct) |  # Stop loss
                (price_change >= take_profit_pct) |  # Take profit
                entry_conditions[i][mask_active]  # New signal
            )
            
            if torch.any(exit_mask):
                position_active[i:][mask_active][exit_mask] = False
                pnl[i][mask_active][exit_mask] = (
                    price_change[exit_mask] * position_size_pct * equity[i-1]
                )

        # Update equity
        equity[i] = equity[i-1] + pnl[i]

    def _calculate_metrics(self, 
                         pnl: torch.Tensor,
                         equity: torch.Tensor,
                         initial_capital: float) -> Dict[str, torch.Tensor]:
        """Calcola metriche della simulazione"""
        try:
            total_trades = torch.sum(pnl != 0, dim=0)
            winning_trades = torch.sum(pnl > 0, dim=0)
            
            metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": winning_trades.float() / total_trades.float(),
                "total_pnl": equity[-1] - initial_capital,
                "final_capital": equity[-1],
                "max_drawdown": self._calculate_max_drawdown(equity),
                "sharpe_ratio": self._calculate_sharpe(equity),
                "profit_factor": self._calculate_profit_factor(pnl)
            }

            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return self._get_empty_metrics(pnl.device, initial_capital)

    def _calculate_max_drawdown(self, equity: torch.Tensor) -> torch.Tensor:
        """Calcola drawdown massimo"""
        try:
            peaks = torch.maximum.accumulate(equity)
            drawdowns = (peaks - equity) / peaks
            return torch.max(drawdowns)
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return torch.tensor(0.0, device=equity.device)

    def _calculate_sharpe(self, equity: torch.Tensor) -> torch.Tensor:
        """Calcola Sharpe ratio"""
        try:
            # Carica fattore di annualizzazione dal config
            annualization = self.config.get("simulator.metrics.annualization_factor", 252)
            
            returns = (equity[1:] - equity[:-1]) / equity[:-1]
            if len(returns) == 0:
                return torch.tensor(0.0, device=equity.device)
            
            std = torch.std(returns)
            if std == 0:
                return torch.tensor(0.0, device=equity.device)
                
            return torch.sqrt(torch.tensor(float(annualization))) * (torch.mean(returns) / std)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return torch.tensor(0.0, device=equity.device)

    def _calculate_profit_factor(self, pnl: torch.Tensor) -> torch.Tensor:
        """Calcola profit factor"""
        try:
            profits = torch.sum(torch.where(pnl > 0, pnl, torch.tensor(0.0, device=pnl.device)))
            losses = torch.abs(torch.sum(torch.where(pnl < 0, pnl, torch.tensor(0.0, device=pnl.device))))
            
            return profits / losses if losses != 0 else torch.tensor(0.0, device=pnl.device)
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return torch.tensor(0.0, device=pnl.device)

    def _get_empty_metrics(self, device: torch.device, initial_capital: float) -> Dict[str, torch.Tensor]:
        """Restituisce metriche vuote per gestione errori"""
        return {
            "total_trades": torch.tensor(0, device=device),
            "winning_trades": torch.tensor(0, device=device),
            "win_rate": torch.tensor(0.0, device=device),
            "total_pnl": torch.tensor(0.0, device=device),
            "final_capital": torch.tensor(initial_capital, device=device),
            "max_drawdown": torch.tensor(0.0, device=device),
            "sharpe_ratio": torch.tensor(0.0, device=device),
            "profit_factor": torch.tensor(0.0, device=device)
        }
