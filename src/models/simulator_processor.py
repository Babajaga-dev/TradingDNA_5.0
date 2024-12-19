import torch
import logging
from typing import Dict, Any

from .position_manager import PositionManager
from .risk_manager import RiskManager
from .trade_stats import TradeStats

logger = logging.getLogger(__name__)

class SimulationProcessor:
    """Processa la simulazione di trading utilizzando operazioni vettorizzate"""
    
    def __init__(self, device_manager, config):
        """
        Inizializza il processor
        
        Args:
            device_manager: Manager per il device di computazione
            config: Configurazione con i parametri
        """
        self.device_manager = device_manager
        self.config = config
        
        # Trading parameters
        self.initial_capital = config.get("simulator.initial_capital", 10000)
        
        # Managers
        self.position_manager = PositionManager(config)
        self.risk_manager = RiskManager(config)
        
        # Computation type
        self.compute_dtype = torch.float32
        
        logger.info(f"Initialized SimulationProcessor with:")
        logger.info(f"Initial capital: ${self.initial_capital}")
        logger.info(f"Position size: {self.position_manager.position_size_pct:.1%}")
        logger.info(f"Max positions: {self.position_manager.max_positions}")
        logger.info(f"Stop loss: {self.position_manager.stop_loss_pct:.1%}")
        logger.info(f"Take profit: {self.position_manager.take_profit_pct:.1%}")
        
    def run_simulation(
        self,
        prices: torch.Tensor,
        entry_conditions: torch.Tensor,
        initial_capital: float
    ) -> Dict[str, torch.Tensor]:
        """
        Esegue la simulazione di trading
        
        Args:
            prices: Serie storica dei prezzi
            entry_conditions: Segnali di entrata
            initial_capital: Capitale iniziale
            
        Returns:
            Dict con i risultati della simulazione
        """
        try:
            logger.info("-" * 80)
            logger.info(f"Starting simulation with capital: ${initial_capital}")
            
            # Verifica segnali
            total_signals = torch.sum(entry_conditions).item()
            if total_signals == 0:
                logger.warning("No entry signals found! Returning empty results.")
                return self._get_empty_tensors(len(prices), initial_capital)
                
            logger.info(f"Found {total_signals} total entry signals")
            
            # Setup
            prices = prices.to(self.compute_dtype)
            device = prices.device
            size = len(prices)
            
            # Inizializza tensori
            position_tensors = self.position_manager.initialize_position_tensors(
                size=size,
                device=device,
                dtype=self.compute_dtype
            )
            
            # Output tensors
            output = self._initialize_output_tensors(size, device, initial_capital)
            
            # Stato corrente
            current_equity = initial_capital
            
            # Processa timestep per timestep
            for t in range(size):
                # Gestisci posizioni esistenti
                if t > 0:
                    # Copia stato precedente
                    position_tensors["active_positions"][t:t+1] = position_tensors["active_positions"][t-1:t]
                    position_tensors["entry_prices"][t:t+1] = position_tensors["entry_prices"][t-1:t]
                    position_tensors["position_sizes"][t:t+1] = position_tensors["position_sizes"][t-1:t]
                
                # Verifica posizioni attive
                active_mask = position_tensors["active_positions"][t:t+1]  # Mantiene la dimensione temporale
                if torch.any(active_mask):
                    # Calcola variazioni
                    price_changes = self._calculate_price_changes(
                        current_price=prices[t],
                        entry_prices=position_tensors["entry_prices"][t:t+1],  # Mantiene la dimensione temporale
                        active_mask=active_mask
                    )
                    
                    # Log solo se ci sono cambiamenti significativi nel prezzo
                    if logger.isEnabledFor(logging.DEBUG):
                        price_diff = abs(prices[t] - prices[t-1]).item() if t > 0 else 0
                        if price_diff > 0.01:  # Log solo se il prezzo cambia più di 0.01
                            logger.debug(
                                f"t={t} - Price: ${prices[t]:.2f} ({price_diff:+.2f}), "
                                f"Active: {torch.sum(active_mask).item()}, "
                                f"Changes: {price_changes[active_mask].tolist()}"
                            )
                    
                    # Verifica condizioni chiusura
                    max_drawdown = self.risk_manager.check_max_drawdown(current_equity, initial_capital)
                    close_positions = self.position_manager.check_close_conditions(
                        price_changes=price_changes,
                        active_positions=active_mask,
                        max_drawdown_hit=max_drawdown
                    )
                    
                    # Chiudi posizioni
                    if torch.any(close_positions):
                        pnl_result = self.position_manager.calculate_pnl(
                            price_changes=price_changes,
                            position_sizes=position_tensors["position_sizes"][t:t+1],  # Mantiene la dimensione temporale
                            close_mask=close_positions,
                            initial_capital=initial_capital
                        )
                        
                        # Aggiorna equity e PnL
                        total_pnl = pnl_result.sum().item()
                        current_equity += total_pnl
                        output["pnl"][t] = total_pnl
                        
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"t={t} - Closed positions: {torch.sum(close_positions).item()}, "
                                f"PnL: ${total_pnl:+.2f}, Equity: ${current_equity:.2f}"
                            )
                        
                        # Azzera posizioni chiuse
                        position_tensors["active_positions"][t:t+1][close_positions] = False
                        position_tensors["entry_prices"][t:t+1][close_positions] = 0
                        position_tensors["position_sizes"][t:t+1][close_positions] = 0
                
                # Gestisci nuove posizioni
                if current_equity > 0 and t < len(entry_conditions):
                    # Verifica segnale di entrata
                    if entry_conditions[t]:
                        # Verifica posizioni disponibili
                        active_count = position_tensors["active_positions"][t:t+1].sum().item()
                        if active_count < self.position_manager.max_positions:
                            # Trova il primo slot libero
                            free_slots = (~position_tensors["active_positions"][t:t+1]).nonzero(as_tuple=True)[1]  # Usa l'indice 1 per gli slot
                            if len(free_slots) > 0:
                                slot_idx = free_slots[0]
                                # Verifica validità della posizione
                                if self.risk_manager.validate_position_size(
                                    self.position_manager.position_size_pct, 
                                    current_equity
                                ):
                                    # Apri nuova posizione
                                    position_tensors["active_positions"][t:t+1, slot_idx] = True
                                    position_tensors["entry_prices"][t:t+1, slot_idx] = prices[t]
                                    position_tensors["position_sizes"][t:t+1, slot_idx] = self.position_manager.position_size_pct
                                    
                                    if logger.isEnabledFor(logging.DEBUG):
                                        logger.debug(
                                            f"t={t} - New position: price=${prices[t]:.2f}, "
                                            f"size={self.position_manager.position_size_pct:.1%}"
                                        )
                                else:
                                    if logger.isEnabledFor(logging.DEBUG):
                                        logger.debug(
                                            f"t={t} - Invalid position size for equity ${current_equity:.2f}"
                                        )
                
                # Aggiorna output tensors
                output["position_active"][t] = position_tensors["active_positions"][t]  # Copia direttamente la riga delle posizioni
                output["entry_prices"][t] = position_tensors["entry_prices"][t:t+1].sum()
                output["position_sizes"][t] = position_tensors["position_sizes"][t:t+1].sum()
                output["equity"][t] = current_equity
            
            # Calcola statistiche finali
            stats = TradeStats.calculate_stats(
                pnl=output["pnl"],
                equity=output["equity"],
                position_active=output["position_active"],
                initial_capital=initial_capital
            )
            
            # Log risultati
            self._log_simulation_results(stats)
            
            return output
            
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            logger.exception(e)
            return self._get_empty_tensors(len(prices), initial_capital)
            
    def _calculate_price_changes(
        self,
        current_price: torch.Tensor,
        entry_prices: torch.Tensor,
        active_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calcola le variazioni percentuali dei prezzi"""
        price_changes = torch.zeros_like(entry_prices)
        if torch.any(active_mask):
            # Calcola variazioni percentuali solo per le posizioni attive
            valid_entries = (entry_prices != 0) & active_mask
            if torch.any(valid_entries):
                # Calcola direttamente senza expand_as
                price_changes[valid_entries] = (current_price - entry_prices[valid_entries]) / entry_prices[valid_entries]
        return price_changes
            
    def _initialize_output_tensors(
        self,
        size: int,
        device: torch.device,
        initial_capital: float
    ) -> Dict[str, torch.Tensor]:
        """Inizializza i tensori di output"""
        return {
            "position_active": torch.zeros((size, self.position_manager.max_positions), dtype=torch.bool, device=device),
            "entry_prices": torch.zeros(size, dtype=self.compute_dtype, device=device),
            "position_sizes": torch.zeros(size, dtype=self.compute_dtype, device=device),
            "pnl": torch.zeros(size, dtype=self.compute_dtype, device=device),
            "equity": torch.ones(size, dtype=self.compute_dtype, device=device) * initial_capital
        }
        
    def _get_empty_tensors(self, size: int, initial_capital: float) -> Dict[str, torch.Tensor]:
        """Restituisce tensori vuoti in caso di errore"""
        device = self.device_manager.device
        return self._initialize_output_tensors(size, device, initial_capital)
        
    def _log_simulation_results(self, stats: Dict[str, Any]):
        """Log dei risultati della simulazione"""
        logger.info("-" * 80)
        logger.info(f"Simulation completed:")
        logger.info(f"\nPerformance:")
        logger.info(f"- Initial capital: ${self.initial_capital:.2f}")
        logger.info(f"- Final equity: ${stats['final_equity']:.2f}")
        logger.info(f"- Total P&L: ${stats['total_pl']:.2f} ({stats['pl_percentage']:.2f}%)")
        logger.info(f"- Max drawdown: {stats['max_drawdown']:.2f}%")
        logger.info(f"\nTrades:")
        logger.info(f"- Total trades: {stats['total_trades']}")
        logger.info(f"- Winning trades: {stats['winning_trades']}")
        logger.info(f"- Losing trades: {stats['losing_trades']}")
        logger.info(f"- Win rate: {stats['win_rate']:.1f}%")
        logger.info(f"- Average win: ${stats['avg_win']:.2f}")
        logger.info(f"- Average loss: ${stats['avg_loss']:.2f}")
        logger.info(f"- Profit factor: {stats['profit_factor']:.2f}")
        logger.info(f"- Max concurrent positions: {stats['max_concurrent_positions']}")
