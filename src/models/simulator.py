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
        
        for timeperiod in [20, 50]:
            upper, middle, lower = talib.BBANDS(
                self.market_state.close, timeperiod=timeperiod
            )
            self.indicators_cache[f"BB_UPPER_{timeperiod}"] = upper
            self.indicators_cache[f"BB_MIDDLE_{timeperiod}"] = middle
            self.indicators_cache[f"BB_LOWER_{timeperiod}"] = lower

    def run_simulation_vectorized(self, entry_conditions: np.ndarray) -> Dict:
        self._reset_simulation()
        
        actions = np.zeros(len(self.market_state.close))
        actions[entry_conditions] = 1  # Segnali di entrata
        
        # Aggiungi segnali di uscita dopo ogni entrata
        for i in range(1, len(actions)):
            if actions[i-1] == 1:
                actions[i] = -1
        
        position_sizes = np.zeros_like(self.market_state.close)
        equity = np.zeros_like(self.market_state.close)
        equity[0] = self.initial_capital
        
        position_active = False
        entry_price = 0
        
        for i in range(1, len(actions)):
            if actions[i] == 1 and not position_active:
                position_active = True
                entry_price = self.market_state.close[i]
                position_sizes[i] = (equity[i-1] * 0.05) / entry_price
                equity[i] = equity[i-1]
            
            elif actions[i] == -1 and position_active:
                position_active = False
                pnl = position_sizes[i-1] * (self.market_state.close[i] - entry_price)
                equity[i] = equity[i-1] + pnl
                position_sizes[i] = 0
            
            else:
                position_sizes[i] = position_sizes[i-1]
                if position_active:
                    unrealized_pnl = position_sizes[i] * (self.market_state.close[i] - entry_price)
                    equity[i] = equity[i-1] + (unrealized_pnl - unrealized_pnl)
                else:
                    equity[i] = equity[i-1]

        trades = np.sum(np.diff(position_sizes != 0)) // 2
        if trades == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "final_capital": self.initial_capital,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }

        trade_pnls = np.diff(equity)[np.diff(position_sizes != 0) != 0]
        winning_trades = np.sum(trade_pnls > 0)
        
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / peaks
        max_drawdown = np.max(drawdowns)
        
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        
        sharpe = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if len(returns) > 0 and np.std(returns) > 0 else 0.0

        return {
            "total_trades": int(trades),
            "winning_trades": int(winning_trades),
            "win_rate": winning_trades / trades if trades > 0 else 0,
            "total_pnl": float(equity[-1] - self.initial_capital),
            "final_capital": float(equity[-1]),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe)
        }





    def _reset_simulation(self):
        """Reset simulation state"""
        self.positions = []
        self.equity_curve = []