from typing import List, Dict
from enum import Enum
import numpy as np
import talib
from src.models.common import Signal, SignalType, MarketData

class Operator(Enum):
    GREATER = ">"
    LESS = "<"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"

class TradingGene:
    def __init__(self):
        self.available_indicators = {
            "SMA": talib.SMA,
            "EMA": talib.EMA,
            "RSI": talib.RSI,
            "MACD": talib.MACD,
            "BB_UPPER": lambda x, p: talib.BBANDS(x, p)[0],
            "BB_LOWER": lambda x, p: talib.BBANDS(x, p)[2],
            "CLOSE": lambda x: x
        }

        self.dna = {
            # Indicatori di entrata
            "entry_indicator1": "SMA",
            "entry_indicator1_params": {"timeperiod": 20},
            "entry_indicator2": "SMA",
            "entry_indicator2_params": {"timeperiod": 50},
            "entry_operator": Operator.CROSS_ABOVE,
            
            # Indicatori di uscita
            "exit_indicator1": "RSI",
            "exit_indicator1_params": {"timeperiod": 14},
            "exit_indicator2": "CLOSE",
            "exit_indicator2_params": {},
            "exit_operator": Operator.GREATER,
            
            # Parametri di risk management
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "position_size_pct": 1.0
        }
        
        self.fitness_score = None
        self.performance_history = {}

    def mutate(self, mutation_rate: float = 0.1):
        """Muta il DNA del gene"""
        for key in self.dna:
            if np.random.random() < mutation_rate:
                if key.endswith('_params'):
                    for param in self.dna[key]:
                        self.dna[key][param] *= np.random.uniform(0.8, 1.2)
                elif key.endswith('_indicator1') or key.endswith('_indicator2'):
                    self.dna[key] = np.random.choice(list(self.available_indicators.keys()))
                elif key.endswith('_operator'):
                    self.dna[key] = np.random.choice(list(Operator))
                elif key.endswith('_pct'):
                    self.dna[key] *= np.random.uniform(0.8, 1.2)

    def calculate_indicator(self, prices: np.ndarray, indicator: str, params: Dict) -> np.ndarray:
        """Calcola un indicatore sui prezzi"""
        indicator_func = self.available_indicators[indicator]
        return indicator_func(prices, **params)

    def check_condition(self, data1: np.ndarray, operator: Operator, data2: np.ndarray) -> bool:
        """Verifica una condizione tra due serie di dati"""
        if operator == Operator.GREATER:
            return data1[-1] > data2[-1]
        elif operator == Operator.LESS:
            return data1[-1] < data2[-1]
        elif operator == Operator.CROSS_ABOVE:
            return data1[-2] <= data2[-2] and data1[-1] > data2[-1]
        elif operator == Operator.CROSS_BELOW:
            return data1[-2] >= data2[-2] and data1[-1] < data2[-1]
        return False

    def generate_signals(self, market_data: List[MarketData]) -> List[Signal]:
        """Genera segnali basati sulle condizioni del gene"""
        signals = []
        
        if len(market_data) < 50:  # Minimo di dati necessari
            return signals

        # Converti i dati in array numpy per gli indicatori
        prices = np.array([d.close for d in market_data])
        
        # Calcola gli indicatori di entrata
        entry_ind1 = self.calculate_indicator(
            prices, 
            self.dna["entry_indicator1"], 
            self.dna["entry_indicator1_params"]
        )
        entry_ind2 = self.calculate_indicator(
            prices,
            self.dna["entry_indicator2"],
            self.dna["entry_indicator2_params"]
        )
        
        # Calcola gli indicatori di uscita
        exit_ind1 = self.calculate_indicator(
            prices,
            self.dna["exit_indicator1"],
            self.dna["exit_indicator1_params"]
        )
        exit_ind2 = self.calculate_indicator(
            prices,
            self.dna["exit_indicator2"],
            self.dna["exit_indicator2_params"]
        )
        
        # Gestisce i NaN
        if np.isnan(entry_ind1[-1]) or np.isnan(entry_ind2[-1]) or \
           np.isnan(exit_ind1[-1]) or np.isnan(exit_ind2[-1]):
            return signals

        # Verifica condizioni di entrata
        if self.check_condition(entry_ind1, self.dna["entry_operator"], entry_ind2):
            current_price = market_data[-1].close
            stop_loss = current_price * (1 - self.dna["stop_loss_pct"]/100)
            take_profit = current_price * (1 + self.dna["take_profit_pct"]/100)
            
            signals.append(Signal(
                type=SignalType.LONG,
                timestamp=market_data[-1].timestamp,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={"gene_id": id(self)}
            ))
        
        # Verifica condizioni di uscita
        elif self.check_condition(exit_ind1, self.dna["exit_operator"], exit_ind2):
            signals.append(Signal(
                type=SignalType.EXIT,
                timestamp=market_data[-1].timestamp,
                price=market_data[-1].close,
                metadata={"gene_id": id(self)}
            ))
            
        return signals

    def crossover(self, other: 'TradingGene') -> 'TradingGene':
        """Esegue il crossover con un altro gene"""
        child = TradingGene()
        for key in self.dna:
            if np.random.random() < 0.5:
                child.dna[key] = self.dna[key]
            else:
                child.dna[key] = other.dna[key]
        return child