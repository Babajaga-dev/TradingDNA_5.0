from typing import List, Dict
from datetime import datetime
import numpy as np
import pandas as pd
from src.models.common import Signal, SignalType, MarketData, TimeFrame
from src.utils.config import config

class Position:
    def __init__(self, signal: Signal, entry_price: float, size: float):
        self.entry_signal = signal
        self.entry_price = entry_price
        self.entry_time = signal.timestamp
        self.size = size
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0
        self.status = "OPEN"
        
    def close(self, exit_price: float, exit_time: datetime):
        self.exit_price = exit_price
        self.exit_time = exit_time
        if self.entry_signal.type == SignalType.LONG:
            self.pnl = (self.exit_price - self.entry_price) * self.size
        else:  # SHORT
            self.pnl = (self.entry_price - self.exit_price) * self.size
        self.status = "CLOSED"




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

    def get_available_capital(self) -> float:
        """Calcola il capitale disponibile per nuove posizioni"""
        committed_capital = sum(
            pos.size * pos.entry_price 
            for pos in self.positions.values()
        )
        return self.capital - committed_capital
        
    def can_open_position(self, position_size_pct: float, price: float) -> bool:
        """
        Verifica se è possibile aprire una nuova posizione
        
        Args:
            position_size_pct: Percentuale del capitale da utilizzare per la posizione
            price: Prezzo di entrata
        """
        # Calcola il capitale necessario per la nuova posizione
        position_capital = self.capital * (position_size_pct / 100)
        
        # Verifica il capitale disponibile
        available_capital = self.get_available_capital()
        
        # Calcola il numero massimo di posizioni basato sulla size percentuale
        max_positions = int(100 / position_size_pct)
        
        return (len(self.positions) < max_positions and 
                available_capital >= position_capital)

    def _process_signal(self, signal: Signal, current_data: MarketData, gene):
        """Processa un segnale di trading con gestione del capitale migliorata"""
        try:
            if signal.type in [SignalType.LONG, SignalType.SHORT]:
                position_size_pct = config.get("trading.position.size_pct", 5.0)
                position_size_pct = min(position_size_pct, 100.0)
                
                if not self.can_open_position(position_size_pct, current_data.close):
                    #print(f"\n[WARNING] Segnale {signal.type} ignorato - Capitale insufficiente o limite posizioni raggiunto")
                    #print(f"  Capitale totale: ${self.capital:.2f}")
                    #print(f"  Capitale disponibile: ${self.get_available_capital():.2f}")
                    #print(f"  Posizioni aperte: {len(self.positions)}")
                    return
                
                # Calcola la size della posizione
                position_capital = self.capital * (position_size_pct / 100)
                units = position_capital / current_data.close
                units = round(units, 4)
                
                # Crea e registra la nuova posizione
                position = Position(signal, current_data.close, units)
                self.position_counter += 1
                self.positions[f"pos_{self.position_counter}"] = position
                
                #print(f"\n[INFO] Nuova posizione aperta:")
                #print(f"  Tipo: {signal.type}")
                #print(f"  Prezzo: ${current_data.close:.2f}")
                #print(f"  Size: {units:.4f}")
                #print(f"  Capitale impegnato: ${position_capital:.2f}")
            
            elif signal.type == SignalType.EXIT:
                for pos_id in list(self.positions.keys()):
                    self.close_position(pos_id, current_data.close, current_data.timestamp)
                    
        except Exception as e:
            #print(f"\nError processing signal: {str(e)}")
            raise

    def close_position(self, position_id: str, exit_price: float, exit_time: datetime):
        """Chiude una posizione e aggiorna il capitale"""
        if position_id not in self.positions:
            #print(f"Warning: Tentativo di chiudere posizione inesistente {position_id}")
            return
            
        position = self.positions.pop(position_id)
        position.close(exit_price, exit_time)
        self.historical_positions.append(position)
        
        # Aggiorna il capitale dopo la chiusura della posizione
        self.capital += position.pnl
        self.equity_curve.append((exit_time, self.capital))
        
        #print(f"\n[INFO] Posizione chiusa:")
        #print(f"  ID: {position_id}")
        #print(f"  P&L: ${position.pnl:.2f}")
        #print(f"  Capitale aggiornato: ${self.capital:.2f}")



        
    def add_market_data(self, timeframe: TimeFrame, data: pd.DataFrame):
        """Aggiunge dati di mercato per un specifico timeframe"""
        #print(f"Adding market data for timeframe {timeframe.value}")
        #print(f"Data shape: {data.shape}")
        #print(f"First timestamp: {data['timestamp'].iloc[0]}")
        #print(f"Last timestamp: {data['timestamp'].iloc[-1]}")
        
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
        
        #print(f"Processed {len(self.market_data[timeframe])} candles")
    
    def run_simulation(self, gene):
        """Esegue la simulazione per un specifico gene"""
        min_candles = config.get("simulator.min_candles", 50)
        
        if TimeFrame.M1 not in self.market_data:
            raise ValueError(f"Dati al minuto (M1) non trovati. Timeframes disponibili: {list(self.market_data.keys())}")
            
        if not self.market_data[TimeFrame.M1]:
            raise ValueError("Nessun dato al minuto trovato")
            
        #print(f"\nAvvio simulazione con {len(self.market_data[TimeFrame.M1])} candele")
        
        self.capital = self.initial_capital
        self.positions.clear()
        self.historical_positions.clear()
        self.equity_curve = [(self.market_data[TimeFrame.M1][0].timestamp, self.capital)]
        
        total_candles = len(self.market_data[TimeFrame.M1])
        progress_interval = max(1, total_candles // 20)
        signals_generated = 0
        
        for i, current_data in enumerate(self.market_data[TimeFrame.M1]):
            try:
                self.current_time = current_data.timestamp
                
                if i % progress_interval == 0:
                    progress = (i / total_candles) * 100
                    current_profit = self.capital - self.initial_capital
                    #print(f"\rProgresso: {progress:.1f}% - P&L: ${current_profit:.2f} - "
                    #      f"Trade aperti: {len(self.positions)} - "
                     #     f"Segnali generati: {signals_generated}", end="")
                
                self._update_positions(current_data)
                
                historical_data = [d for d in self.market_data[TimeFrame.M1] 
                                 if d.timestamp <= current_data.timestamp]
                if len(historical_data) < min_candles:  # Minimo di dati necessari per gli indicatori
                    continue
                
                signals = gene.generate_signals(historical_data)
                if signals:
                    signals_generated += len(signals)
                    for signal in signals:
                        self._process_signal(signal, current_data, gene)
                        
            except Exception as e:
                #print(f"\nError during simulation at candle {i}: {str(e)}")
                raise
        
        #print("\nSimulazione completata")
            
    def _update_positions(self, current_data: MarketData):
        """Aggiorna lo stato delle posizioni aperte"""
        for pos_id, position in list(self.positions.items()):
            try:
                if position.entry_signal.stop_loss:
                    if (position.entry_signal.type == SignalType.LONG and 
                        current_data.low <= position.entry_signal.stop_loss):
                        self.close_position(pos_id, position.entry_signal.stop_loss, 
                                         current_data.timestamp)
                    elif (position.entry_signal.type == SignalType.SHORT and 
                          current_data.high >= position.entry_signal.stop_loss):
                        self.close_position(pos_id, position.entry_signal.stop_loss, 
                                         current_data.timestamp)
                
                if position.entry_signal.take_profit:
                    if (position.entry_signal.type == SignalType.LONG and 
                        current_data.high >= position.entry_signal.take_profit):
                        self.close_position(pos_id, position.entry_signal.take_profit, 
                                         current_data.timestamp)
                    elif (position.entry_signal.type == SignalType.SHORT and 
                          current_data.low <= position.entry_signal.take_profit):
                        self.close_position(pos_id, position.entry_signal.take_profit, 
                                         current_data.timestamp)
            except Exception as e:
                #print(f"\nError updating position {pos_id}: {str(e)}")
                raise
    
    def get_performance_metrics(self) -> Dict:
        """Calcola le metriche di performance della simulazione"""
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
        
        # Calcola Sharpe Ratio (semplificato)
        returns = np.diff([e[1] for e in self.equity_curve]) / \
                 [e[1] for e in self.equity_curve][:-1]
        if len(returns) > 0:
            sharpe = np.sqrt(252) * (np.mean(returns) / np.std(returns)) \
                    if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl / 10,  # Dividiamo anche qui il P&L totale per 10
            "final_capital": self.capital,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe
        }
