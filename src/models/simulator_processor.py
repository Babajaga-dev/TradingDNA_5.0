import torch
import logging
import traceback
import math
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SimulationProcessor:
    def __init__(self, device_manager, config):
        self.device_manager = device_manager
        self.config = config

        # Trading parameters
        self.initial_capital = config.get("simulator.initial_capital", 10000)
        self.position_size_pct = config.get("trading.position.size_pct", 5) / 100
        self.stop_loss_pct = config.get("trading.position.stop_loss_pct", 1.5) / 100
        self.take_profit_pct = config.get("trading.position.take_profit_pct", 3.0) / 100

        # Calculate max_positions based on size
        self.max_positions = math.floor(1 / self.position_size_pct)

        # Risk limits
        self.max_drawdown_pct = 0.15

        # Computation type
        self.compute_dtype = torch.float32

        logger.info(f"Initialized SimulationProcessor with:")
        logger.info(f"Initial capital: ${self.initial_capital}")
        logger.info(f"Position size: {self.position_size_pct*100}%")
        logger.info(f"Max positions: {self.max_positions}")
        logger.info(f"Stop loss: {self.stop_loss_pct*100}%")
        logger.info(f"Take profit: {self.take_profit_pct*100}%")
        logger.info(f"Max drawdown: {self.max_drawdown_pct*100}%")

    def run_simulation(self, prices: torch.Tensor, entry_conditions: torch.Tensor, initial_capital: float) -> Dict[str, torch.Tensor]:
        """Run the simulation using vectorized operations."""
        try:
            logger.info(f"Starting simulation with capital: ${initial_capital}")

            total_signals = torch.sum(entry_conditions).item()
            if total_signals == 0:
                logger.warning("No entry signals found! Returning empty results.")
                return self._get_empty_tensors(len(prices), initial_capital)

            logger.info(f"Found {total_signals} total entry signals")

            prices = prices.to(self.compute_dtype)
            device = prices.device
            size = len(prices)

            position_active = torch.zeros(size, dtype=torch.bool, device=device)
            entry_prices = torch.zeros(size, dtype=self.compute_dtype, device=device)
            position_sizes = torch.zeros(size, dtype=self.compute_dtype, device=device)
            pnl = torch.zeros(size, dtype=self.compute_dtype, device=device)
            equity = torch.ones(size, dtype=self.compute_dtype, device=device) * initial_capital

            active_positions = torch.zeros(size, self.max_positions, dtype=torch.bool, device=device)
            position_entries = torch.zeros(size, self.max_positions, dtype=self.compute_dtype, device=device)
            position_sizes_matrix = torch.zeros(size, self.max_positions, dtype=self.compute_dtype, device=device)

            current_equity = initial_capital

            batch_size = 5000
            for start_idx in range(0, size, batch_size):
                end_idx = min(start_idx + batch_size, size)

                if start_idx > 0:
                    active_positions[start_idx:end_idx] = active_positions[start_idx-1].clone()
                    position_entries[start_idx:end_idx] = position_entries[start_idx-1].clone()
                    position_sizes_matrix[start_idx:end_idx] = position_sizes_matrix[start_idx-1].clone()

                    curr_prices = prices[start_idx:end_idx].unsqueeze(1)
                    price_changes = torch.where(
                        position_entries[start_idx:end_idx] > 0,
                        (curr_prices - position_entries[start_idx:end_idx]) / position_entries[start_idx:end_idx],
                        torch.zeros_like(curr_prices)
                    )

                    stop_loss = price_changes <= -self.stop_loss_pct
                    take_profit = price_changes >= self.take_profit_pct
                    max_drawdown = equity[start_idx-1] < initial_capital * (1 - self.max_drawdown_pct)

                    close_positions = active_positions[start_idx:end_idx] & (stop_loss | take_profit | max_drawdown)

                    position_pnl = torch.where(
                        close_positions,
                        price_changes * position_sizes_matrix[start_idx:end_idx] * current_equity,
                        torch.zeros_like(price_changes)
                    )

                    total_pnl = position_pnl.sum(dim=1)
                    pnl[start_idx:end_idx] += total_pnl
                    current_equity += total_pnl.sum().item()
                    equity[start_idx:end_idx] = current_equity

                    active_positions[start_idx:end_idx] &= ~close_positions
                    position_entries[start_idx:end_idx][close_positions] = 0
                    position_sizes_matrix[start_idx:end_idx][close_positions] = 0

                batch_entries = entry_conditions[start_idx:end_idx]
                can_open = active_positions[start_idx:end_idx].sum(dim=1) < self.max_positions
                valid_entries = batch_entries & can_open

                if torch.any(valid_entries):
                    for t in range(len(valid_entries)):
                        if valid_entries[t]:
                            slot_idx = (~active_positions[start_idx + t]).nonzero(as_tuple=True)[0][0]
                            active_positions[start_idx + t, slot_idx] = True
                            position_entries[start_idx + t, slot_idx] = prices[start_idx + t]
                            position_sizes_matrix[start_idx + t, slot_idx] = self.position_size_pct

                unrealized_changes = torch.where(
                    position_entries[start_idx:end_idx] > 0,
                    (prices[start_idx:end_idx].unsqueeze(1) - position_entries[start_idx:end_idx]) / position_entries[start_idx:end_idx],
                    torch.zeros_like(position_entries[start_idx:end_idx])
                )

                unrealized_pnl = unrealized_changes * position_sizes_matrix[start_idx:end_idx] * current_equity
                total_unrealized = unrealized_pnl.sum(dim=1)
                equity[start_idx:end_idx] = current_equity + total_unrealized

                position_active[start_idx:end_idx] = active_positions[start_idx:end_idx].any(dim=1)
                entry_prices[start_idx:end_idx] = position_entries[start_idx:end_idx].sum(dim=1)
                position_sizes[start_idx:end_idx] = position_sizes_matrix[start_idx:end_idx].sum(dim=1)

            max_concurrent = torch.max(active_positions.sum(dim=1)).item()
            total_trades = torch.sum(pnl != 0).item()
            final_equity = equity[-1].item()
            min_equity = torch.min(equity).item()

            logger.info(f"Simulation completed with final equity: ${final_equity:.2f}, total trades: {total_trades}, max concurrent positions: {max_concurrent}")

            return {
                "position_active": position_active,
                "entry_prices": entry_prices,
                "position_sizes": position_sizes,
                "pnl": pnl,
                "equity": equity
            }

        except Exception as e:
            logger.error(f"Error in simulation: {e}")
            logger.error(traceback.format_exc())
            return self._get_empty_tensors(len(prices), initial_capital)

    def _get_empty_tensors(self, size: int, initial_capital: float) -> Dict[str, torch.Tensor]:
        device = self.device_manager.device
        dtype = self.device_manager.dtype
        return {
            "position_active": torch.zeros(size, dtype=torch.bool, device=device),
            "entry_prices": torch.zeros(size, dtype=dtype, device=device),
            "position_sizes": torch.zeros(size, dtype=dtype, device=device),
            "pnl": torch.zeros(size, dtype=dtype, device=device),
            "equity": torch.ones(size, dtype=dtype, device=device) * initial_capital
        }
