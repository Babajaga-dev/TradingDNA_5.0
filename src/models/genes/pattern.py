# src/models/genes/pattern.py
import numpy as np
import talib
from typing import List, Dict

from .base import TradingGene, GeneType
from ..common import Signal, SignalType, MarketData
from ...utils.config import config

class PatternRecognitionGene(TradingGene):
    def __init__(self, random_init=True):
        super().__init__(random_init=False)
        self.gene_type = GeneType.PATTERN.value
        
        self.pattern_name_map = {
            "ENGULFING": "CDLENGULFING",
            "HAMMER": "CDLHAMMER",
            "DOJI": "CDLDOJI",
            "EVENINGSTAR": "CDLEVENINGSTAR",
            "MORNINGSTAR": "CDLMORNINGSTAR",
            "HARAMI": "CDLHARAMI",
            "SHOOTINGSTAR": "CDLSHOOTINGSTAR",
            "MARUBOZU": "CDLMARUBOZU"
        }
        
        enabled_patterns = config.get("trading.pattern_gene.patterns", [])
        self.available_patterns = {}
        
        for pattern in enabled_patterns:
            if pattern in self.pattern_name_map:
                talib_name = self.pattern_name_map[pattern]
                if hasattr(talib, talib_name):
                    self.available_patterns[pattern] = (
                        getattr(talib, talib_name),
                        2 if pattern in ["ENGULFING", "EVENINGSTAR", "MORNINGSTAR"] else 1
                    )
        
        params = config.get("trading.pattern_gene.parameters", {})
        
        if random_init:
            self.initialize_pattern_dna(params)
        else:
            self.dna.update({
                "required_patterns": params.get("required_patterns", {}).get("default", 2),
                "pattern_window": params.get("pattern_window", {}).get("default", 3),
                "confirmation_periods": params.get("confirmation_periods", {}).get("default", 1)
            })

    def initialize_pattern_dna(self, params):
        self.dna.update({
            "required_patterns": np.random.randint(
                params.get("required_patterns", {}).get("min", 1),
                params.get("required_patterns", {}).get("max", 3)
            ),
            "pattern_window": np.random.randint(
                params.get("pattern_window", {}).get("min", 2),
                params.get("pattern_window", {}).get("max", 5)
            ),
            "confirmation_periods": np.random.randint(
                params.get("confirmation_periods", {}).get("min", 1),
                params.get("confirmation_periods", {}).get("max", 3)
            )
        })

    def detect_patterns(self, market_data: List[MarketData]) -> Dict[str, int]:
        if len(market_data) < 10:
            return {}
            
        opens = np.array([d.open for d in market_data])
        highs = np.array([d.high for d in market_data])
        lows = np.array([d.low for d in market_data])
        closes = np.array([d.close for d in market_data])
        
        patterns = {}
        for name, (func, _) in self.available_patterns.items():
            result = func(opens, highs, lows, closes)
            if not np.isnan(result[-1]):
                patterns[name] = result[-1]
                
        return patterns

    def generate_signals(self, market_data: List[MarketData]) -> List[Signal]:
        signals = super().generate_signals(market_data)
        
        if signals and signals[0].type in [SignalType.LONG, SignalType.SHORT]:
            patterns = self.detect_patterns(market_data)
            bullish_patterns = sum(1 for v in patterns.values() if v > 0)
            bearish_patterns = sum(1 for v in patterns.values() if v < 0)
            
            signal_type = signals[0].type
            if (signal_type == SignalType.LONG and 
                bullish_patterns < self.dna["required_patterns"]):
                return []
            elif (signal_type == SignalType.SHORT and 
                  bearish_patterns < self.dna["required_patterns"]):
                return []
                
        return signals
