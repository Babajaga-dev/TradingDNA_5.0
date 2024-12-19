import torch
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class PositionManager:
    """Gestisce l'apertura, chiusura e tracking delle posizioni"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza il position manager
        
        Args:
            config: Configurazione con i parametri di trading
        """
        # Trading parameters
        self.position_size_pct = config.get("trading.position.size_pct", 0.05)  # default 5%
        self.max_positions = self._calculate_max_positions()
        
        # Risk parameters
        self.stop_loss_pct = config.get("trading.position.stop_loss_pct", 0.015)  # default 1.5%
        self.take_profit_pct = config.get("trading.position.take_profit_pct", 0.03)  # default 3.0%
        
        logger.debug(f"Initialized PositionManager:")
        logger.debug(f"Position size: {self.position_size_pct:.1%}")
        logger.debug(f"Max positions: {self.max_positions}")
        logger.debug(f"Stop loss: {self.stop_loss_pct:.1%}")
        logger.debug(f"Take profit: {self.take_profit_pct:.1%}")
        
    def _calculate_max_positions(self) -> int:
        """Calcola il numero massimo di posizioni basato sulla position size"""
        return max(1, int(1 / self.position_size_pct))
        
    def initialize_position_tensors(self, size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """
        Inizializza i tensori per il tracking delle posizioni
        
        Args:
            size: Numero di timesteps
            device: Device su cui allocare i tensori
            dtype: Tipo di dato per i tensori
            
        Returns:
            Dict con i tensori inizializzati
        """
        return {
            "active_positions": torch.zeros(size, self.max_positions, dtype=torch.bool, device=device),
            "entry_prices": torch.zeros(size, self.max_positions, dtype=dtype, device=device),
            "position_sizes": torch.zeros(size, self.max_positions, dtype=dtype, device=device)
        }
        
    def check_close_conditions(
        self,
        price_changes: torch.Tensor,
        active_positions: torch.Tensor,
        max_drawdown_hit: bool
    ) -> torch.Tensor:
        """
        Verifica le condizioni di chiusura per le posizioni attive
        
        Args:
            price_changes: Variazioni percentuali dei prezzi
            active_positions: Maschera delle posizioni attive
            max_drawdown_hit: Flag che indica se è stato raggiunto il max drawdown
            
        Returns:
            Maschera booleana delle posizioni da chiudere
        """
        # Verifica solo posizioni attive
        if not torch.any(active_positions):
            return torch.zeros_like(active_positions)
            
        # Verifica solo posizioni attive
        close_mask = torch.zeros_like(active_positions)
        
        # Applica le condizioni solo alle posizioni attive
        active_indices = torch.nonzero(active_positions)
        for idx in active_indices:
            t, slot = idx[0], idx[1]
            # Verifica stop loss e take profit
            if price_changes[t, slot] <= -self.stop_loss_pct or \
               price_changes[t, slot] >= self.take_profit_pct or \
               max_drawdown_hit:
                close_mask[t, slot] = True
        
        # Debug log
        if torch.any(close_mask):
            logger.debug(f"Close conditions triggered:")
            logger.debug(f"Active positions: {active_positions.tolist()}")
            logger.debug(f"Price changes: {price_changes.tolist()}")
            logger.debug(f"Close mask: {close_mask.tolist()}")
            
            # Log dettagliato per ogni posizione chiusa
            closed_positions = torch.nonzero(close_mask)
            for pos in closed_positions:
                t, slot = pos[0].item(), pos[1].item()
                logger.debug(f"Closing position at t={t}, slot={slot}:")
                logger.debug(f"Price change: {price_changes[t, slot]:.2%}")
                if price_changes[t, slot] <= -self.stop_loss_pct:
                    logger.debug("Reason: Stop Loss")
                elif price_changes[t, slot] >= self.take_profit_pct:
                    logger.debug("Reason: Take Profit")
                elif max_drawdown_hit:
                    logger.debug("Reason: Max Drawdown")
        
        return close_mask
        
    def calculate_pnl(
        self,
        price_changes: torch.Tensor,
        position_sizes: torch.Tensor,
        close_mask: torch.Tensor,
        initial_capital: float
    ) -> torch.Tensor:
        """
        Calcola il P&L per le posizioni chiuse
        
        Args:
            price_changes: Variazioni percentuali dei prezzi
            position_sizes: Size delle posizioni
            close_mask: Maschera delle posizioni da chiudere
            initial_capital: Capitale iniziale
            
        Returns:
            P&L calcolato per le posizioni chiuse
        """
        # Calcola P&L solo per le posizioni che stiamo chiudendo
        pnl = torch.zeros_like(price_changes)
        if torch.any(close_mask):
            # Calcola il valore in dollari delle posizioni usando le variazioni già in decimale
            pct_changes = price_changes.clone()
            position_value = position_sizes * initial_capital
            
            # Applica la maschera e calcola il P&L
            pct_changes.masked_fill_(~close_mask, 0)
            position_value.masked_fill_(~close_mask, 0)
            pnl = pct_changes * position_value
            
            # Debug log
            closed_positions = torch.nonzero(close_mask)
            for pos in closed_positions:
                t, slot = pos[0].item(), pos[1].item()
                logger.debug(f"PnL calculation for position at t={t}, slot={slot}:")
                logger.debug(f"Price change: {price_changes[t, slot]:.2%}")
                logger.debug(f"Position size: ${position_value[t, slot]:.2f}")
                logger.debug(f"PnL: ${pnl[t, slot]:.2f}")
            
        return pnl
        
    def find_entry_slots(
        self,
        entry_signals: torch.Tensor,
        active_positions: torch.Tensor,
        current_equity: float,
        initial_capital: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trova gli slot disponibili per nuove posizioni
        
        Args:
            entry_signals: Segnali di entrata
            active_positions: Maschera delle posizioni attive
            current_equity: Equity corrente
            initial_capital: Capitale iniziale
            
        Returns:
            Tuple con (indici delle righe, indici degli slot) per le nuove posizioni
        """
        device = entry_signals.device
        
        # Verifica capitale disponibile
        position_value = self.position_size_pct * initial_capital
        if position_value < 1.0:  # Minimo $1 per posizione
            logger.debug("Insufficient capital for new positions")
            return torch.tensor([], dtype=torch.long, device=device), \
                   torch.tensor([], dtype=torch.long, device=device)
        
        # Verifica segnali di entrata
        entry_rows = entry_signals.nonzero(as_tuple=True)[0]
        if len(entry_rows) == 0:
            return torch.tensor([], dtype=torch.long, device=device), \
                   torch.tensor([], dtype=torch.long, device=device)
        
        # Verifica posizioni disponibili
        active_count = active_positions.sum(dim=1)
        can_open = active_count < self.max_positions
        
        # Filtra solo le righe dove possiamo aprire posizioni
        valid_rows = []
        slot_indices = []
        
        for row_idx in entry_rows:
            if can_open[row_idx]:
                # Trova il primo slot libero
                free_slots = (~active_positions[row_idx]).nonzero(as_tuple=True)[0]
                if len(free_slots) > 0:
                    valid_rows.append(row_idx.item())
                    slot_indices.append(free_slots[0].item())
                    
                    # Debug log
                    logger.debug(f"Found entry slot:")
                    logger.debug(f"Row: {row_idx.item()}")
                    logger.debug(f"Slot: {free_slots[0].item()}")
                    logger.debug(f"Active positions: {active_positions[row_idx].sum().item()}")
                    logger.debug(f"Position value: ${position_value:.2f}")
        
        if not valid_rows:
            return torch.tensor([], dtype=torch.long, device=device), \
                   torch.tensor([], dtype=torch.long, device=device)
                   
        return torch.tensor(valid_rows, dtype=torch.long, device=device), \
               torch.tensor(slot_indices, dtype=torch.long, device=device)
