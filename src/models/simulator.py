# src/models/simulator.py
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import talib
import logging

from .common import Signal, SignalType, MarketData, TimeFrame, Position
from ..utils.config import config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    
)

logger = logging.getLogger(__name__)

class TradingSimulator:
    def __init__(self):
        self.initial_capital = config.get("simulator.initial_capital", 10000)
        self.capital = self.initial_capital
        self.min_candles = config.get("simulator.min_candles", 50)
        self.positions: Dict[str, Position] = {}
        self.historical_positions: List[Position] = []
        self.market_data: Dict[TimeFrame, List[MarketData]] = {}
        self.current_time = None
        self.position_counter = 0
        self.equity_curve = []
        self.gene_performance = {}
        self.indicators_cache = {}

    def add_market_data(self, timeframe: TimeFrame, data: pd.DataFrame):
        """Aggiunge dati di mercato e precalcola gli indicatori"""
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
        
        self._precalculate_indicators()

    def _precalculate_indicators(self):
        """Precalcola gli indicatori comuni per ottimizzare la simulazione"""
        if TimeFrame.M1 not in self.market_data:
            logger.warning("No M1 data found for precalculation")
            return

        prices = np.array([d.close for d in self.market_data[TimeFrame.M1]])
        highs = np.array([d.high for d in self.market_data[TimeFrame.M1]])
        lows = np.array([d.low for d in self.market_data[TimeFrame.M1]])
        opens = np.array([d.open for d in self.market_data[TimeFrame.M1]])
        
        logger.info("Precalculating common indicators...")
        
        # Common SMA periods
        common_periods = [5, 8, 13, 21, 34, 55, 89]
        for period in common_periods:
            cache_key = f"SMA_{period}"
            self.indicators_cache[cache_key] = talib.SMA(prices, timeperiod=period)
            
        # Common RSI periods
        rsi_periods = [7, 14, 21]
        for period in rsi_periods:
            cache_key = f"RSI_{period}"
            self.indicators_cache[cache_key] = talib.RSI(prices, timeperiod=period)

        # Common Bollinger Bands periods
        bb_periods = [20, 50]
        for period in bb_periods:
            upper, middle, lower = talib.BBANDS(prices, timeperiod=period)
            self.indicators_cache[f"BB_UPPER_{period}"] = upper
            self.indicators_cache[f"BB_MIDDLE_{period}"] = middle 
            self.indicators_cache[f"BB_LOWER_{period}"] = lower

        # Common MACD parameters
        self.indicators_cache["MACD"] = talib.MACD(prices)[0]  # Solo linea MACD

        # Common Candlestick Patterns
        for pattern in ["CDLENGULFING", "CDLHAMMER", "CDLDOJI", "CDLHARAMI", "CDLMARUBOZU"]:
            if hasattr(talib, pattern):
                self.indicators_cache[pattern] = getattr(talib, pattern)(
                    opens, highs, lows, prices
                )

        logger.info("Indicators precalculation completed")

    def get_cached_indicator(self, indicator_type: str, params: Dict) -> Optional[np.ndarray]:
        """Recupera indicatore dalla cache se disponibile"""
        try:
            if indicator_type == "SMA":
                period = params.get("timeperiod")
                cache_key = f"SMA_{period}"
                return self.indicators_cache.get(cache_key)
                    
            elif indicator_type == "RSI":
                period = params.get("timeperiod")
                cache_key = f"RSI_{period}"
                return self.indicators_cache.get(cache_key)
                    
            elif indicator_type.startswith("BB_"):
                period = params.get("timeperiod")
                cache_key = f"{indicator_type}_{period}"
                return self.indicators_cache.get(cache_key)
                    
            elif indicator_type == "MACD":
                return self.indicators_cache.get("MACD")
                    
            elif indicator_type.startswith("CDL"):
                return self.indicators_cache.get(indicator_type)
                
        except Exception as e:
            logger.error(f"Error getting cached indicator {indicator_type}: {e}")
            
        return None

    def run_simulation(self, gene):
        """Esegue simulazione trading con cache indicatori"""
        if TimeFrame.M1 not in self.market_data:
            raise ValueError("M1 timeframe data not found")
            
        self._reset_simulation()
        
        for i, current_data in enumerate(self.market_data[TimeFrame.M1]):
            if i < self.min_candles:
                continue
                
            self.current_time = current_data.timestamp
            self._update_positions(current_data)
            
            historical_data = self.market_data[TimeFrame.M1][:i+1]
            signals = self._generate_signals_optimized(gene, historical_data)
            
            if signals:
                for signal in signals:
                    self._process_signal(signal, current_data, gene)
            
            # Update equity curve
            total_value = self.capital + sum(
                pos.size * current_data.close for pos in self.positions.values()
            )
            self.equity_curve.append((current_data.timestamp, total_value))

    def _generate_signals_optimized(self, gene, market_data: List[MarketData]) -> List[Signal]:
        """Genera segnali utilizzando indicatori precalcolati quando possibile"""
        try:
            # First try to get cached indicators
            entry_ind1 = self.get_cached_indicator(
                gene.dna["entry_indicator1"],
                gene.dna["entry_indicator1_params"]
            )
            entry_ind2 = self.get_cached_indicator(
                gene.dna["entry_indicator2"],
                gene.dna["entry_indicator2_params"]
            )
            exit_ind1 = self.get_cached_indicator(
                gene.dna["exit_indicator1"],
                gene.dna["exit_indicator1_params"]
            )
            exit_ind2 = self.get_cached_indicator(
                gene.dna["exit_indicator2"],
                gene.dna["exit_indicator2_params"]
            )
            
            # If not in cache, calculate them
            if entry_ind1 is None:
                entry_ind1 = gene.calculate_indicators(
                    market_data,
                    gene.dna["entry_indicator1"],
                    gene.dna["entry_indicator1_params"]
                )
            if entry_ind2 is None:
                entry_ind2 = gene.calculate_indicators(
                    market_data,
                    gene.dna["entry_indicator2"], 
                    gene.dna["entry_indicator2_params"]
                )
            if exit_ind1 is None:
                exit_ind1 = gene.calculate_indicators(
                    market_data,
                    gene.dna["exit_indicator1"],
                    gene.dna["exit_indicator1_params"]
                )
            if exit_ind2 is None:
                exit_ind2 = gene.calculate_indicators(
                    market_data,
                    gene.dna["exit_indicator2"],
                    gene.dna["exit_indicator2_params"]
                )
            
            # Generate signals using the indicators
            return gene.generate_signals(market_data)
            
        except Exception as e:
            logger.error(f"Error in optimized signal generation: {e}")
            return []

    def _reset_simulation(self):
        """Resetta lo stato della simulazione"""
        self.capital = self.initial_capital
        self.positions.clear()
        self.historical_positions.clear()
        self.equity_curve = []
        if self.market_data.get(TimeFrame.M1):
            self.equity_curve.append(
                (self.market_data[TimeFrame.M1][0].timestamp, self.capital)
            )

    def _process_signal(self, signal: Signal, current_data: MarketData, gene=None):
        """Processa un segnale di trading"""
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
        """Aggiorna le posizioni aperte"""
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
        """Chiude una posizione"""
        if position_id not in self.positions:
            return
            
        position = self.positions.pop(position_id)
        position.close(exit_price, exit_time)
        self.historical_positions.append(position)
        
        self.capital += position.pnl
        self.equity_curve.append((exit_time, self.capital))
        
        if position.signal.metadata and 'gene_id' in position.signal.metadata:
            gene_id = position.signal.metadata['gene_id']
            if gene_id not in self.gene_performance:
                self.gene_performance[gene_id] = []
            self.gene_performance[gene_id].append(position.pnl)

    def can_open_position(self, position_size_pct: float, price: float) -> bool:
        """Verifica se è possibile aprire una nuova posizione"""
        position_capital = self.capital * (position_size_pct / 100)
        available_capital = self.get_available_capital()
        max_positions = int(100 / position_size_pct)
        
        return (len(self.positions) < max_positions and 
                available_capital >= position_capital)

    def get_available_capital(self) -> float:
        """Calcola il capitale disponibile"""
        committed_capital = sum(pos.size * pos.entry_price 
                              for pos in self.positions.values())
        return self.capital - committed_capital

    def get_performance_metrics(self) -> Dict:
        """Calcola le metriche di performance"""
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