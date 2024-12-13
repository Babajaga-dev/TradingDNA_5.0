import torch
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

    def process_batch(self, batch_slice: slice, prices: torch.Tensor,
                    entry_conditions: torch.Tensor, position_active: torch.Tensor,
                    entry_prices: torch.Tensor, pnl: torch.Tensor,
                    equity: torch.Tensor, initial_capital: float) -> None:
        """
        Processa un batch di dati con controlli sui valori
        """
        try:
            # Gestione memoria prima del processing
            if self.device_manager.memory_config["cache_mode"] == "all":
                self.device_manager.manage_memory()
                
            # Setup stream per processing asincrono
            stream = self.device_manager.get_stream(batch_slice.start % max(1, len(self.device_manager.streams)))
            
            with stream:
                with torch.amp.autocast('cuda') if self.device_manager.mixed_precision else nullcontext():
                    self._process_batch_data(
                        batch_slice, prices, entry_conditions,
                        position_active, entry_prices, pnl,
                        equity, initial_capital
                    )
                    
            # Sincronizza stream se necessario
            if isinstance(stream, torch.cuda.Stream):
                stream.synchronize()
                    
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            logger.error(traceback.format_exc())

    def _process_batch_data(self, batch_slice: slice, prices: torch.Tensor,
                         entry_conditions: torch.Tensor, position_active: torch.Tensor,
                         entry_prices: torch.Tensor, pnl: torch.Tensor,
                         equity: torch.Tensor, initial_capital: float) -> None:
        """Elabora i dati del batch con controlli di validità"""
        # Calcoli base con controlli di validità
        current_prices = prices[batch_slice]
        current_conditions = entry_conditions[batch_slice]
        
        # Ottieni lo stato precedente in modo sicuro
        prev_idx = max(0, batch_slice.start - 1)
        prev_active = position_active[prev_idx]

        # Verifica valori non finiti nei prezzi
        if torch.any(~torch.isfinite(current_prices)):
            prev_price = prices[prev_idx]
            current_prices = torch.nan_to_num(current_prices, nan=prev_price)
        
        # Processa nuove entrate
        new_entries = torch.logical_and(current_conditions, ~prev_active)
        position_active[batch_slice] = new_entries
        
        # Aggiorna prezzi di entrata con controllo
        valid_prices = torch.where(
            torch.isfinite(current_prices),
            current_prices,
            entry_prices[batch_slice]
        )
        entry_prices[batch_slice] = torch.where(new_entries, valid_prices, entry_prices[batch_slice])
        
        # Processa uscite se necessario
        if prev_active:
            self._process_exits(
                batch_slice, current_prices, current_conditions,
                position_active, entry_prices, pnl,
                equity, initial_capital, prices[prev_idx]
            )

    def _process_exits(self, batch_slice: slice, current_prices: torch.Tensor,
                     current_conditions: torch.Tensor, position_active: torch.Tensor,
                     entry_prices: torch.Tensor, pnl: torch.Tensor,
                     equity: torch.Tensor, initial_capital: float,
                     prev_price: torch.Tensor) -> None:
        """Gestisce le uscite dalle posizioni"""
        prev_idx = max(0, batch_slice.start - 1)
        entry_price = entry_prices[prev_idx]
        
        if not torch.isfinite(entry_price) or entry_price == 0:
            entry_price = prev_price
        
        # Calcola variazioni prezzo con protezione da divisione per zero
        safe_entry = torch.where(entry_price == 0, 
                            torch.ones_like(entry_price) * prev_price, 
                            entry_price)
        price_changes = (current_prices - safe_entry) / safe_entry
        
        # Verifica valori non finiti nelle variazioni
        price_changes = torch.nan_to_num(price_changes, nan=0.0)
        
        # Condizioni di uscita
        stops = price_changes <= -self.stop_loss_pct
        targets = price_changes >= self.take_profit_pct
        
        exits = torch.logical_or(
            torch.logical_or(stops, targets),
            current_conditions
        )
        
        # Aggiorna posizioni
        position_active[batch_slice] = torch.logical_and(
            position_active[batch_slice],
            ~exits
        )
        
        # Calcola PnL con controllo valori
        current_equity = equity[prev_idx]
        if not torch.isfinite(current_equity) or current_equity <= 0:
            current_equity = torch.tensor(initial_capital, 
                                    device=self.device_manager.device, 
                                    dtype=self.device_manager.dtype)
        
        batch_pnl = torch.where(
            exits,
            price_changes * self.position_size_pct * current_equity,
            torch.zeros_like(price_changes)
        )
        
        # Verifica valori PnL
        batch_pnl = torch.nan_to_num(batch_pnl, nan=0.0)
        pnl[batch_slice] = batch_pnl
        
        # Aggiorna equity con controllo
        self._update_equity(batch_slice, batch_pnl, equity, initial_capital)

    def _update_equity(self, batch_slice: slice, current_pnl: torch.Tensor,
                     equity: torch.Tensor, initial_capital: float) -> None:
        """Aggiorna l'equity con controlli di validità"""
        if not torch.all(torch.isfinite(current_pnl)):
            current_pnl = torch.nan_to_num(current_pnl, nan=0.0)
        
        prev_idx = max(0, batch_slice.start - 1)
        previous_equity = equity[prev_idx]
        if not torch.isfinite(previous_equity) or previous_equity <= 0:
            previous_equity = torch.tensor(initial_capital, 
                                    device=self.device_manager.device, 
                                    dtype=self.device_manager.dtype)
        
        new_equity = previous_equity + current_pnl
        
        # Assicura che l'equity non scenda sotto zero
        new_equity = torch.maximum(
            new_equity,
            torch.tensor(1e-6, device=self.device_manager.device, dtype=self.device_manager.dtype)
        )
        
        equity[batch_slice] = new_equity

    def run_simulation(self, prices: torch.Tensor, entry_conditions: torch.Tensor,
                     initial_capital: float) -> Dict[str, torch.Tensor]:
        """Esegue la simulazione completa"""
        try:
            # Inizializza arrays
            position_active = torch.zeros_like(prices, dtype=torch.bool, device=self.device_manager.device)
            entry_prices = torch.zeros_like(prices, device=self.device_manager.device)
            pnl = torch.zeros_like(prices, device=self.device_manager.device)
            equity = torch.ones_like(prices, device=self.device_manager.device) * initial_capital
            
            # Determina batch size ottimale
            batch_size = self.device_manager.get_optimal_batch_size(len(prices))
            
            # Processa in chunks paralleli
            chunk_size = self.device_manager.parallel_config["chunk_size"]
            num_chunks = (len(prices) + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, len(prices))
                
                # Processa batch all'interno del chunk
                for i in range(chunk_start, chunk_end, batch_size):
                    end_idx = min(i + batch_size, chunk_end)
                    
                    self.process_batch(
                        slice(i, end_idx),
                        prices, entry_conditions,
                        position_active, entry_prices,
                        pnl, equity, initial_capital
                    )
                
                # Gestione memoria dopo ogni chunk
                if self.device_manager.memory_config["cache_mode"] != "all":
                    self.device_manager.manage_memory()

            return {
                "position_active": position_active,
                "entry_prices": entry_prices,
                "pnl": pnl,
                "equity": equity
            }
                
        except Exception as e:
            logger.error("Error in simulation:")
            logger.error(traceback.format_exc())
            raise
