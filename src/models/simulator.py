# src/models/simulator.py
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import talib
import logging
from dataclasses import dataclass

from .common import Signal, SignalType, MarketData, TimeFrame, Position
from ..utils.config import config

logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

class TradingSimulator:
    def __init__(self):
        self.initial_capital = config.get("simulator.initial_capital", 10000)
        self.min_candles = config.get("simulator.min_candles", 50)
        self.indicators_cache = {}
        self.market_state = None
        self.positions = []
        self.equity_curve = []
        self.metrics = None

    def add_market_data(self, timeframe: TimeFrame, data: pd.DataFrame):
        """Converte dati in arrays numpy per processamento veloce"""
        self.market_state = MarketState(
            timestamp=data.timestamp.values,
            open=data.open.values.astype(np.float64),
            high=data.high.values.astype(np.float64),
            low=data.low.values.astype(np.float64),
            close=data.close.values.astype(np.float64),
            volume=data.volume.values.astype(np.float64)
        )
        self._precalculate_indicators()

    def _precalculate_indicators(self):
        """Precalcola indicatori comuni"""
        logger.info("Precalculating common indicators...")
        
        periods = [5, 8, 13, 21, 34, 55, 89]
        for period in periods:
            self.indicators_cache[f"SMA_{period}"] = talib.SMA(
                self.market_state.close, timeperiod=period
            )
            self.indicators_cache[f"EMA_{period}"] = talib.EMA(
                self.market_state.close, timeperiod=period
            )
            self.indicators_cache[f"RSI_{period}"] = talib.RSI(
                self.market_state.close, timeperiod=period
            )
            
            upper, middle, lower = talib.BBANDS(
                self.market_state.close, timeperiod=period
            )
            self.indicators_cache[f"BB_UPPER_{period}"] = upper
            self.indicators_cache[f"BB_MIDDLE_{period}"] = middle
            self.indicators_cache[f"BB_LOWER_{period}"] = lower

    def run_simulation_vectorized(self, entry_conditions: np.ndarray) -> Dict:
        """Esegue simulazione vettorizzata e salva metriche"""
        self._reset_simulation()
        
        position_size_pct = config.get("trading.position.size_pct", 5) / 100
        stop_loss_pct = config.get("trading.position.stop_loss_pct", 2) / 100
        take_profit_pct = config.get("trading.position.take_profit_pct", 4) / 100
        
        prices = self.market_state.close
        position_active = np.zeros_like(prices, dtype=bool)
        entry_prices = np.zeros_like(prices)
        pnl = np.zeros_like(prices)
        equity = np.ones_like(prices) * self.initial_capital
        trade_results = []
        
        for i in range(1, len(prices)):
            current_price = prices[i]
            
            if entry_conditions[i] and not position_active[i-1]:
                # Entry
                position_active[i:] = True
                entry_prices[i:] = current_price
                
            elif position_active[i-1]:
                # Check exit conditions
                entry_price = entry_prices[i-1]
                price_change = (current_price - entry_price) / entry_price
                
                should_exit = (
                    price_change <= -stop_loss_pct or  # Stop loss
                    price_change >= take_profit_pct or  # Take profit
                    entry_conditions[i]  # New signal
                )
                
                if should_exit:
                    position_active[i:] = False
                    pnl[i] = price_change * position_size_pct * equity[i-1]
                    trade_results.append(pnl[i])
                    
            # Update equity
            equity[i] = equity[i-1] + pnl[i]
        
        # Calculate metrics
        total_trades = len(trade_results)
        if total_trades == 0:
            self.metrics = {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "final_capital": self.initial_capital,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "profit_factor": 0
            }
        else:
            winning_trades = sum(1 for x in trade_results if x > 0)
            
            # Drawdown calculation
            peaks = np.maximum.accumulate(equity)
            drawdowns = (peaks - equity) / peaks
            max_drawdown = np.max(drawdowns)
            
            # Returns and Sharpe calculation
            returns = np.diff(equity) / equity[:-1]
            returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
            
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.sqrt(252) * (np.mean(returns) / np.std(returns))
            else:
                sharpe = 0
                
            # Profit Factor
            gross_profits = sum(x for x in trade_results if x > 0)
            gross_losses = abs(sum(x for x in trade_results if x < 0))
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else 0
            
            self.metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": winning_trades / total_trades,
                "total_pnl": float(equity[-1] - self.initial_capital),
                "final_capital": float(equity[-1]),
                "max_drawdown": float(max_drawdown),
                "sharpe_ratio": float(sharpe),
                "profit_factor": float(profit_factor)
            }
            
        return self.metrics

    def _reset_simulation(self):
        """Reset simulation state"""
        self.positions = []
        self.equity_curve = []
        self.metrics = None

    def get_performance_metrics(self) -> Dict:
        """Returns current performance metrics"""
        if self.metrics is None:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "final_capital": self.initial_capital,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "profit_factor": 0
            }
        return self.metrics