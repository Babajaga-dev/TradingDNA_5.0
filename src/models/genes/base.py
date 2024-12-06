# src/models/genes/base.py
from enum import Enum
from typing import Dict, List
import numpy as np
from datetime import datetime
import logging
import traceback

from ..common import Signal, SignalType, MarketData
from ...utils.config import config
from .indicators import IndicatorRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class Operator(Enum):
    GREATER = ">"
    LESS = "<"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"

class TradingGene:
    def __init__(self, random_init=True):
        logger.info("Inizializzazione TradingGene")
        try:
            self.fitness_score = None
            self.performance_history = {}
            self.mutation_history = []
            self.available_indicators = IndicatorRegistry.get_available_indicators()
            
            if random_init:
                self._initialize_random()
            else:
                self._load_from_config()
                
            logger.info("TradingGene inizializzato con successo")
        except Exception as e:
            logger.error(f"Errore inizializzazione TradingGene: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _initialize_random(self):
        logger.debug("Inizializzazione random DNA")
        try:
            indicators = list(self.available_indicators.keys())
            operators = [op.value for op in Operator]
            
            self.dna = {
                "entry_indicator1": np.random.choice(indicators),
                "entry_indicator1_params": {},
                "entry_indicator2": np.random.choice(indicators),
                "entry_indicator2_params": {},
                "entry_operator": np.random.choice(operators),
                "exit_indicator1": np.random.choice(indicators),
                "exit_indicator1_params": {},
                "exit_indicator2": np.random.choice(indicators),
                "exit_indicator2_params": {},
                "exit_operator": np.random.choice(operators),
                "stop_loss_pct": np.random.uniform(0.5, 5.0),
                "take_profit_pct": np.random.uniform(1.0, 10.0),
                "position_size_pct": np.random.uniform(1.0, 20.0)
            }
            
            # Genera parametri per ogni indicatore
            for key in ["entry_indicator1", "entry_indicator2", "exit_indicator1", "exit_indicator2"]:
                indicator = self.dna[key]
                ind_params = self.available_indicators[indicator]["params"]
                self.dna[f"{key}_params"] = IndicatorRegistry.generate_random_params(ind_params)
                
        except Exception as e:
            logger.error(f"Errore in _initialize_random: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _load_from_config(self):
        logger.debug("Caricamento configurazione")
        try:
            self.dna = {
                "entry_indicator1": config.get("trading.indicators.entry.indicator1.type", "SMA"),
                "entry_indicator1_params": config.get("trading.indicators.entry.indicator1.params", {"timeperiod": 20}),
                "entry_indicator2": config.get("trading.indicators.entry.indicator2.type", "SMA"),
                "entry_indicator2_params": config.get("trading.indicators.entry.indicator2.params", {"timeperiod": 50}),
                "entry_operator": config.get("trading.indicators.entry.operator", "cross_above"),
                "exit_indicator1": config.get("trading.indicators.exit.indicator1.type", "RSI"),
                "exit_indicator1_params": config.get("trading.indicators.exit.indicator1.params", {"timeperiod": 14}),
                "exit_indicator2": config.get("trading.indicators.exit.indicator2.type", "CLOSE"),
                "exit_indicator2_params": config.get("trading.indicators.exit.indicator2.params", {}),
                "exit_operator": config.get("trading.indicators.exit.operator", ">"),
                "stop_loss_pct": config.get("trading.position.stop_loss_pct", 2.0),
                "take_profit_pct": config.get("trading.position.take_profit_pct", 4.0),
                "position_size_pct": config.get("trading.position.size_pct", 5.0)
            }
        except Exception as e:
            logger.error(f"Errore in _load_from_config: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def calculate_indicators(self, market_data: List[MarketData], indicator: str, params: Dict) -> np.ndarray:
        try:
            if len(market_data) < 2:
                return np.array([np.nan] * 2)
                
            prices = np.array([d.close for d in market_data])
            result = IndicatorRegistry.calculate_indicator(
                prices, indicator, params, self.available_indicators, market_data
            )
            
            if not isinstance(result, np.ndarray):
                logger.error(f"Indicatore {indicator} ha restituito tipo non valido: {type(result)}")
                return np.array([np.nan] * len(market_data))
                
            return result
            
        except Exception as e:
            logger.error(f"Errore in calculate_indicators per {indicator}: {str(e)}")
            logger.error(traceback.format_exc())
            return np.array([np.nan] * len(market_data))

    def check_condition(self, values1: np.ndarray, values2: np.ndarray, operator: str) -> bool:
        try:
            if len(values1) < 2 or len(values2) < 2:
                return False
                
            if np.isnan(values1[-1]) or np.isnan(values2[-1]):
                return False
                
            if operator == Operator.GREATER.value:
                return values1[-1] > values2[-1]
            elif operator == Operator.LESS.value:
                return values1[-1] < values2[-1]
            elif operator == Operator.CROSS_ABOVE.value:
                return values1[-2] <= values2[-2] and values1[-1] > values2[-1]
            elif operator == Operator.CROSS_BELOW.value:
                return values1[-2] >= values2[-2] and values1[-1] < values2[-1]
            return False
            
        except Exception as e:
            logger.error(f"Errore in check_condition: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _get_max_lookback(self, params: Dict) -> int:
        """Get maximum lookback period needed for indicator parameters"""
        lookback = 1
        for param in params.values():
            if isinstance(param, (int, float)) and param > lookback:
                lookback = int(param)
        return lookback + 5  # Add buffer

    def _validate_indicator_values(self, values: np.ndarray, indicator_name: str) -> bool:
        """Validate indicator values are usable"""
        if values is None or len(values) < 2:
            logger.debug(f"{indicator_name} returned insufficient data")
            return False
            
        if np.isnan(values[-2:]).any():
            logger.debug(f"{indicator_name} contains NaN values")
            return False
            
        if np.isinf(values[-2:]).any():
            logger.debug(f"{indicator_name} contains Inf values")
            return False
            
        return True

    def generate_signals(self, market_data: List[MarketData]) -> List[Signal]:
        try:
            # Check minimum required data points
            min_lookback = max(
                self._get_max_lookback(self.dna["entry_indicator1_params"]),
                self._get_max_lookback(self.dna["entry_indicator2_params"]),
                self._get_max_lookback(self.dna["exit_indicator1_params"]),
                self._get_max_lookback(self.dna["exit_indicator2_params"])
            )
            
            min_candles = config.get("simulator.min_candles", 50)
            if len(market_data) < max(min_lookback + 2, min_candles):
                logger.debug(f"Insufficient data points: {len(market_data)} < {max(min_lookback + 2, min_candles)}")
                return []

            current_data = market_data[-1]
            
            # Calculate entry indicators with validation
            entry_ind1 = self.calculate_indicators(
                market_data[-min_lookback:], 
                self.dna["entry_indicator1"], 
                self.dna["entry_indicator1_params"]
            )
            entry_ind2 = self.calculate_indicators(
                market_data[-min_lookback:],
                self.dna["entry_indicator2"],
                self.dna["entry_indicator2_params"]
            )
            
            # Validate entry indicators
            if not self._validate_indicator_values(entry_ind1, "entry_indicator1"):
                return []
            if not self._validate_indicator_values(entry_ind2, "entry_indicator2"):
                return []

            # Entry conditions
            if self.check_condition(entry_ind1, entry_ind2, self.dna["entry_operator"]):
                stop_loss = current_data.close * (1 - self.dna["stop_loss_pct"] / 100)
                take_profit = current_data.close * (1 + self.dna["take_profit_pct"] / 100)
                
                return [Signal(
                    type=SignalType.LONG,
                    timestamp=current_data.timestamp,
                    price=current_data.close,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )]
            
            # Calculate and validate exit indicators
            exit_ind1 = self.calculate_indicators(
                market_data[-min_lookback:],
                self.dna["exit_indicator1"],
                self.dna["exit_indicator1_params"]
            )
            exit_ind2 = self.calculate_indicators(
                market_data[-min_lookback:],
                self.dna["exit_indicator2"],
                self.dna["exit_indicator2_params"]
            )
            
            if not self._validate_indicator_values(exit_ind1, "exit_indicator1"):
                return []
            if not self._validate_indicator_values(exit_ind2, "exit_indicator2"):
                return []
                
            if self.check_condition(exit_ind1, exit_ind2, self.dna["exit_operator"]):
                return [Signal(
                    type=SignalType.EXIT,
                    timestamp=current_data.timestamp,
                    price=current_data.close
                )]

            return []
            
        except Exception as e:
            logger.error(f"Error in generate_signals: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def mutate(self, mutation_rate: float = 0.1):
        try:
            mutations = []
            for key in self.dna:
                if np.random.random() < mutation_rate:
                    if key.endswith('_params'):
                        indicator_key = key.replace('_params', '')
                        if indicator_key in self.dna:
                            indicator = self.dna[indicator_key]
                            params = self.available_indicators[indicator]["params"]
                            self.dna[key] = IndicatorRegistry.generate_random_params(params)
                    elif key.startswith(('entry_indicator', 'exit_indicator')):
                        self.dna[key] = np.random.choice(list(self.available_indicators.keys()))
                    elif key.endswith('_operator'):
                        self.dna[key] = np.random.choice([op.value for op in Operator])
                    elif key.endswith('_pct'):
                        if 'stop_loss' in key:
                            self.dna[key] = np.random.uniform(0.5, 5.0)
                        elif 'take_profit' in key:
                            self.dna[key] = np.random.uniform(1.0, 10.0)
                        elif 'position_size' in key:
                            self.dna[key] = np.random.uniform(1.0, 20.0)
                    mutations.append(key)
                    
            if mutations:
                logger.debug(f"Mutated parameters: {mutations}")
                
        except Exception as e:
            logger.error(f"Errore in mutate: {str(e)}")
            logger.error(traceback.format_exc())

    def crossover(self, other: 'TradingGene') -> 'TradingGene':
        try:
            child = self.__class__(random_init=False)
            for key in self.dna:
                child.dna[key] = self.dna[key] if np.random.random() < 0.5 else other.dna[key]
            return child
        except Exception as e:
            logger.error(f"Errore in crossover: {str(e)}")
            logger.error(traceback.format_exc())
            return self.__class__(random_init=True)