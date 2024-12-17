import torch
import intel_extension_for_pytorch as ipex
import logging
from typing import Dict, Any
from contextlib import nullcontext
import traceback

logger = logging.getLogger(__name__)

class SimulationProcessor:
    def __init__(self, device_manager, config):
        self.device_manager = device_manager
        self.config = config
        
        # Parametri trading
        self.position_size_pct = config.get("trading.position.size_pct", 5) / 100
        self.stop_loss_pct = config.get("trading.position.stop_loss_pct", 2) / 100
        self.take_profit_pct = config.get("trading.position.take_profit_pct", 4) / 100

        # Inizializza compute dtype per calcoli critici
        self.compute_dtype = torch.float32

    def run_simulation(self, prices: torch.Tensor, entry_conditions: torch.Tensor,
                      initial_capital: float) -> Dict[str, torch.Tensor]:
        """Esegue la simulazione completa"""
        try:
            # Converti prezzi a compute_dtype per calcoli precisi
            prices = prices.to(self.compute_dtype)
            
            # Inizializza arrays nel compute_dtype
            position_active = torch.zeros_like(prices, dtype=torch.bool)
            entry_prices = torch.zeros_like(prices, dtype=self.compute_dtype)
            pnl = torch.zeros_like(prices, dtype=self.compute_dtype)
            equity = torch.ones_like(prices, dtype=self.compute_dtype) * initial_capital

            # Determina batch size
            batch_size = max(32, min(1024, len(prices) // 100))
            
            # Process in batches
            for i in range(0, len(prices), batch_size):
                end_idx = min(i + batch_size, len(prices))
                self.process_batch(
                    slice(i, end_idx),
                    prices, entry_conditions,
                    position_active, entry_prices,
                    pnl, equity, initial_capital
                )

            # Converti risultati al dtype originale
            return {
                "position_active": position_active,
                "entry_prices": entry_prices.to(self.device_manager.dtype),
                "pnl": pnl.to(self.device_manager.dtype),
                "equity": equity.to(self.device_manager.dtype)
            }
                
        except Exception as e:
            logger.error(f"Error in simulation: {e}")
            return self._get_empty_tensors(len(prices), initial_capital)

    def process_batch(self, batch_slice: slice, prices: torch.Tensor,
                    entry_conditions: torch.Tensor, position_active: torch.Tensor,
                    entry_prices: torch.Tensor, pnl: torch.Tensor,
                    equity: torch.Tensor, initial_capital: float) -> None:
        """Processa un batch di dati"""
        try:
            # Prendi i dati del batch
            current_prices = prices[batch_slice]
            current_conditions = entry_conditions[batch_slice]
            
            # Ottieni stato precedente
            prev_idx = max(0, batch_slice.start - 1)
            prev_active = position_active[prev_idx]
            prev_price = prices[prev_idx]

            # Calcola nuove entrate
            new_entries = torch.logical_and(current_conditions, ~prev_active)
            position_active[batch_slice] = new_entries

            # Aggiorna prezzi entrata
            valid_prices = torch.where(
                torch.isfinite(current_prices),
                current_prices,
                prev_price
            )
            entry_prices[batch_slice] = torch.where(new_entries, valid_prices, entry_prices[batch_slice])

            # Processa uscite se necessario
            if prev_active:
                self._process_exits(
                    batch_slice, valid_prices, current_conditions,
                    position_active, entry_prices, pnl,
                    equity, initial_capital, prev_price
                )
                
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")

    def _process_exits(self, batch_slice: slice, current_prices: torch.Tensor,
                     current_conditions: torch.Tensor, position_active: torch.Tensor,
                     entry_prices: torch.Tensor, pnl: torch.Tensor,
                     equity: torch.Tensor, initial_capital: float,
                     prev_price: torch.Tensor) -> None:
        """Processa le uscite dalle posizioni"""
        try:
            prev_idx = max(0, batch_slice.start - 1)
            entry_price = entry_prices[prev_idx]
            
            # Calcola variazioni prezzo sicure
            safe_entry = torch.where(
                torch.logical_or(~torch.isfinite(entry_price), entry_price == 0),
                prev_price,
                entry_price
            )
            
            price_changes = (current_prices - safe_entry) / safe_entry
            price_changes = torch.clamp(price_changes, min=-0.5, max=0.5)

            # Condizioni uscita
            stops = price_changes <= -self.stop_loss_pct
            targets = price_changes >= self.take_profit_pct
            exits = torch.logical_or(torch.logical_or(stops, targets), current_conditions)

            # Aggiorna posizioni
            position_active[batch_slice] = torch.logical_and(
                position_active[batch_slice],
                ~exits
            )

            # Calcola PnL sicuro
            current_equity = equity[prev_idx]
            current_equity = torch.where(
                torch.logical_or(~torch.isfinite(current_equity), current_equity <= 0),
                torch.tensor(initial_capital, dtype=self.compute_dtype, device=current_equity.device),
                current_equity
            )

            # Calcola PnL con limiti
            safe_size = min(self.position_size_pct, 0.25)  # Max 25% position size
            batch_pnl = torch.where(
                exits,
                price_changes * safe_size * current_equity,
                torch.zeros_like(price_changes)
            )

            # Limita PnL
            max_pnl = current_equity * 0.1  # Max 10% movimento
            batch_pnl = torch.clamp(batch_pnl, min=-max_pnl, max=max_pnl)
            
            pnl[batch_slice] = batch_pnl
            
            # Aggiorna equity
            self._update_equity(batch_slice, batch_pnl, equity, initial_capital)
            
        except Exception as e:
            logger.error(f"Error processing exits: {e}")

    def _update_equity(self, batch_slice: slice, current_pnl: torch.Tensor,
                     equity: torch.Tensor, initial_capital: float) -> None:
        """Aggiorna equity in modo sicuro"""
        try:
            prev_idx = max(0, batch_slice.start - 1)
            previous_equity = equity[prev_idx]

            # Sanitizza equity precedente
            previous_equity = torch.where(
                torch.logical_or(~torch.isfinite(previous_equity), previous_equity <= 0),
                torch.tensor(initial_capital, dtype=self.compute_dtype, device=previous_equity.device),
                previous_equity
            )

            # Calcola nuova equity con limiti
            new_equity = previous_equity + current_pnl
            min_equity = torch.tensor(1e-3, dtype=self.compute_dtype, device=new_equity.device)
            max_equity = torch.tensor(initial_capital * 5, dtype=self.compute_dtype, device=new_equity.device)
            
            new_equity = torch.clamp(new_equity, min=min_equity, max=max_equity)
            equity[batch_slice] = new_equity
            
        except Exception as e:
            logger.error(f"Error updating equity: {e}")
            # Mantieni equity precedente in caso di errore
            equity[batch_slice] = previous_equity

    def _get_empty_tensors(self, size: int, initial_capital: float) -> Dict[str, torch.Tensor]:
        """Crea tensori vuoti in caso di errore"""
        device = self.device_manager.device
        dtype = self.device_manager.dtype
        return {
            "position_active": torch.zeros(size, dtype=torch.bool, device=device),
            "entry_prices": torch.zeros(size, dtype=dtype, device=device),
            "pnl": torch.zeros(size, dtype=dtype, device=device),
            "equity": torch.ones(size, dtype=dtype, device=device) * initial_capital
        }