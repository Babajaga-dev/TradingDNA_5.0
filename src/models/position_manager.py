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
        trading_config = config.get("trading", {}).get("position", {})
        self.position_size_pct = trading_config.get("size_pct", 0.05)  # default 5%
        self.max_positions = self._calculate_max_positions()
        
        # Risk parameters
        self.stop_loss_pct = trading_config.get("stop_loss_pct", 0.015)  # default 1.5%
        self.take_profit_pct = trading_config.get("take_profit_pct", 0.03)  # default 3.0%
        
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
        Verifica le condizioni di chiusura per le posizioni attive usando operazioni vettorizzate
        
        Args:
            price_changes: Variazioni percentuali dei prezzi
            active_positions: Maschera delle posizioni attive
            max_drawdown_hit: Flag che indica se è stato raggiunto il max drawdown
            
        Returns:
            Maschera booleana delle posizioni da chiudere
        """
        # Se non ci sono posizioni attive o max_drawdown, ritorna subito
        if not torch.any(active_positions) and not max_drawdown_hit:
            return torch.zeros_like(active_positions)
            
        # Crea la maschera di chiusura usando operazioni vettorizzate
        close_mask = torch.zeros_like(active_positions)
        
        # Applica le condizioni solo alle posizioni attive
        if torch.any(active_positions):
            # Verifica stop loss e take profit in modo vettorizzato
            stop_loss_mask = price_changes <= -self.stop_loss_pct
            take_profit_mask = price_changes >= self.take_profit_pct
            
            # Combina le condizioni
            close_mask = (stop_loss_mask | take_profit_mask) & active_positions
            
            # Applica max_drawdown se necessario
            if max_drawdown_hit:
                close_mask = close_mask | active_positions
        
            # Debug log
            if torch.any(close_mask):
                logger.debug("Close conditions triggered:")
                logger.debug(f"Active positions: {torch.sum(active_positions).item()}")
                logger.debug(f"Stop loss hits: {torch.sum(stop_loss_mask & active_positions).item()}")
                logger.debug(f"Take profit hits: {torch.sum(take_profit_mask & active_positions).item()}")
                if max_drawdown_hit:
                    logger.debug("Max drawdown triggered")
        
        return close_mask
        
    def calculate_pnl(
        self,
        price_changes: torch.Tensor,
        position_sizes: torch.Tensor,
        close_mask: torch.Tensor,
        initial_capital: float
    ) -> torch.Tensor:
        """
        Calcola il P&L per le posizioni chiuse usando operazioni vettorizzate
        
        Args:
            price_changes: Variazioni percentuali dei prezzi
            position_sizes: Size delle posizioni
            close_mask: Maschera delle posizioni da chiudere
            initial_capital: Capitale iniziale
            
        Returns:
            P&L calcolato per le posizioni chiuse
        """
        # Se non ci sono posizioni da chiudere, ritorna subito
        if not torch.any(close_mask):
            return torch.zeros_like(price_changes)
            
        # Calcola il P&L in modo vettorizzato
        position_value = position_sizes * initial_capital
        pnl = torch.where(close_mask, price_changes * position_value, torch.zeros_like(price_changes))
        
        # Debug log
        if logger.isEnabledFor(logging.DEBUG):
            total_closed = torch.sum(close_mask).item()
            total_pnl = torch.sum(pnl).item()
            avg_price_change = torch.sum(price_changes * close_mask).item() / total_closed if total_closed > 0 else 0
            logger.debug(f"PnL calculation summary:")
            logger.debug(f"Closed positions: {total_closed}")
            logger.debug(f"Average price change: {avg_price_change:.8%}")
            logger.debug(f"Total PnL: ${total_pnl:.8f}")
            
        return pnl
        
    def find_entry_slots(
        self,
        entry_signals: torch.Tensor,
        active_positions: torch.Tensor,
        current_equity: float,
        initial_capital: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trova gli slot disponibili per nuove posizioni usando operazioni vettorizzate
        
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
        position_value = self.position_size_pct * current_equity
        if position_value < 1.0 or current_equity < initial_capital * 0.1:  # Minimo $1 per posizione e 10% del capitale iniziale
            logger.debug(f"Insufficient capital for new positions (equity: ${current_equity:.8f}, position value: ${position_value:.8f})")
            return torch.tensor([], dtype=torch.long, device=device), \
                   torch.tensor([], dtype=torch.long, device=device)
        
        # Verifica segnali di entrata e posizioni disponibili in modo vettorizzato
        active_count = active_positions.sum(dim=1)
        can_open = (active_count < self.max_positions) & entry_signals
        
        if not torch.any(can_open):
            return torch.tensor([], dtype=torch.long, device=device), \
                   torch.tensor([], dtype=torch.long, device=device)
        
        # Trova le righe valide
        valid_rows = can_open.nonzero(as_tuple=True)[0]
        
        # Trova gli slot liberi per ogni riga valida
        slot_indices = torch.zeros_like(valid_rows)
        for i, row in enumerate(valid_rows):
            # Prendi il primo slot libero
            slot_indices[i] = (~active_positions[row]).nonzero(as_tuple=True)[0][0]
        
        # Debug log
        if len(valid_rows) > 0:
            logger.debug(f"Found {len(valid_rows)} entry slots")
            logger.debug(f"Position value: ${position_value:.8f}")
        
        return valid_rows, slot_indices
