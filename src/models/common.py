# src/models/common.py
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Dict, Optional

class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"

@dataclass
class MarketData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: TimeFrame
    
@dataclass
class Signal:
    type: SignalType
    timestamp: datetime
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None

@dataclass
class Position:
    entry_price: float
    entry_time: datetime
    size: float
    signal: Signal
    status: str = "OPEN"
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0

    def close(self, exit_price: float, exit_time: datetime):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = "CLOSED"
        
        if self.signal.type == SignalType.LONG:
            self.pnl = (self.exit_price - self.entry_price) * self.size
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.size