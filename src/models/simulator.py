# src/models/simulator.py
from typing import Dict, List
import numpy as np
import pandas as pd
from datetime import datetime

from .common import Signal, SignalType, MarketData, TimeFrame, Position
from ..utils.config import config

class TradingSimulator:
    def __init__(self):
        self.initial_capital = config.get("simulator.initial_capital", 10000)
        self.capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.historical_positions: List[Position] = []
        self.market_data: Dict[TimeFrame, List[MarketData]] = {}
        self.current_time = None
        self.position_counter = 0
        self.equity_curve = []
        self.gene_performance = {}

    def add_market_data(self, timeframe: TimeFrame, data: pd.DataFrame):
        self.market_data[timeframe] = []
        for _, row in data.iterrows():
            market_data = MarketData(
                timestamp=row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                timeframe=timeframe
            )
            self.market_data[timeframe].append(market_data)

    def run_simulation(self, gene):
        min_candles = config.get("simulator.min_candles", 50)
        
        if TimeFrame.M1 not in self.market_data:
            raise ValueError("Dati al minuto (M1) non trovati")
            
        self._reset_simulation()
        
        for i, current_data in enumerate(self.market_data[TimeFrame.M1]):
            self.current_time = current_data.timestamp
            self._update_positions(current_data)
            
            historical_data = [d for d in self.market_data[TimeFrame.M1] 
                             if d.timestamp <= current_data.timestamp]
            if len(historical_data) < min_candles:
                continue
                
            signals = gene.generate_signals(historical_data)
            if signals:
                for signal in signals:
                    self._process_signal(signal, current_data, gene)

    def _reset_simulation(self):
        self.capital = self.initial_capital
        self.positions.clear()
        self.historical_positions.clear()
        self.equity_curve = [(self.market_data[TimeFrame.M1][0].timestamp, self.capital)]

    def _process_signal(self, signal: Signal, current_data: MarketData, gene=None):
        if signal.type in [SignalType.LONG, SignalType.SHORT]:
            position_size_pct = config.get("trading.position.size_pct", 5.0)
            if not self.can_open_position(position_size_pct, current_data.close):
                return
                
            position_capital = self.capital * (position_size_pct / 100)
            units = position_capital / current_data.close
            
            position = Position(
                entry_price=current_data.close,
                entry_time=current_data.timestamp,
                size=units,
                signal=signal
            )
            
            self.position_counter += 1
            self.positions[f"pos_{self.position_counter}"] = position
            
        elif signal.type == SignalType.EXIT:
            for pos_id in list(self.positions.keys()):
                self.close_position(pos_id, current_data.close, current_data.timestamp)

    def _update_positions(self, current_data: MarketData):
        for pos_id, position in list(self.positions.items()):
            if position.signal.stop_loss:
                if (position.signal.type == SignalType.LONG and 
                    current_data.low <= position.signal.stop_loss):
                    self.close_position(pos_id, position.signal.stop_loss, 
                                     current_data.timestamp)
                elif (position.signal.type == SignalType.SHORT and 
                      current_data.high >= position.signal.stop_loss):
                    self.close_position(pos_id, position.signal.stop_loss, 
                                     current_data.timestamp)
            
            if position.signal.take_profit:
                if (position.signal.type == SignalType.LONG and 
                    current_data.high >= position.signal.take_profit):
                    self.close_position(pos_id, position.signal.take_profit, 
                                     current_data.timestamp)
                elif (position.signal.type == SignalType.SHORT and 
                      current_data.low <= position.signal.take_profit):
                    self.close_position(pos_id, position.signal.take_profit, 
                                     current_data.timestamp)

    def close_position(self, position_id: str, exit_price: float, exit_time: datetime):
        if position_id not in self.positions:
            return
            
        position = self.positions.pop(position_id)
        position.close(exit_price, exit_time)
        self.historical_positions.append(position)
        
        self.capital += position.pnl
        self.equity_curve.append((exit_time, self.capital))
        
        # Track gene performance
        if position.signal.metadata and 'gene_id' in position.signal.metadata:
            gene_id = position.signal.metadata['gene_id']
            if gene_id not in self.gene_performance:
                self.gene_performance[gene_id] = []
            self.gene_performance[gene_id].append(position.pnl)

    def can_open_position(self, position_size_pct: float, price: float) -> bool:
        position_capital = self.capital * (position_size_pct / 100)
        available_capital = self.get_available_capital()
        max_positions = int(100 / position_size_pct)
        
        return (len(self.positions) < max_positions and 
                available_capital >= position_capital)

    def get_available_capital(self) -> float:
        committed_capital = sum(pos.size * pos.entry_price 
                              for pos in self.positions.values())
        return self.capital - committed_capital

    def get_performance_metrics(self) -> Dict:
        if not self.historical_positions:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "final_capital": self.capital,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }

        total_trades = len(self.historical_positions)
        winning_trades = len([p for p in self.historical_positions if p.pnl > 0])
        total_pnl = sum(p.pnl for p in self.historical_positions)
        
        # Calcola drawdown
        equity = [e[1] for e in self.equity_curve]
        peak = equity[0]
        drawdowns = []
        for value in equity:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Calcola Sharpe Ratio
        returns = np.diff([e[1] for e in self.equity_curve]) / \
                 [e[1] for e in self.equity_curve][:-1]
        sharpe = np.sqrt(252) * (np.mean(returns) / np.std(returns)) \
                if len(returns) > 0 and np.std(returns) > 0 else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "final_capital": self.capital,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe
        }