# src/models/genes/volatility.py
import numpy as np
from typing import List

from .base import TradingGene, GeneType
from .indicators import calculate_atr
from ..common import Signal, SignalType, MarketData
from ...utils.config import config

class VolatilityAdaptiveGene(TradingGene):
    def __init__(self, random_init=True):
        super().__init__(random_init=False)
        self.gene_type = GeneType.VOLATILITY.value
        
        params = config.get("trading.volatility_gene.parameters", {})
        
        if random_init:
            self.dna.update({
                "volatility_timeperiod": np.random.randint(
                    params.get("timeperiod", {}).get("min", 10),
                    params.get("timeperiod", {}).get("max", 50)
                ),
                "volatility_multiplier": np.random.uniform(
                    params.get("multiplier", {}).get("min", 0.5),
                    params.get("multiplier", {}).get("max", 2.0)
                ),
                "base_position_size": np.random.uniform(
                    params.get("base_position_size", {}).get("min", 1.0),
                    params.get("base_position_size", {}).get("max", 10.0)
                )
            })
        else:
            self.dna.update({
                "volatility_timeperiod": params.get("timeperiod", {}).get("default", 14),
                "volatility_multiplier": params.get("multiplier", {}).get("default", 1.0),
                "base_position_size": params.get("base_position_size", {}).get("default", 5.0)
            })

    def calculate_position_size(self, prices: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
        atr = calculate_atr(high, low, prices, self.dna["volatility_timeperiod"])
        if np.isnan(atr[-1]):
            return self.dna["base_position_size"]
            
        avg_price = np.mean(prices[-self.dna["volatility_timeperiod"]:])
        normalized_atr = atr[-1] / avg_price
        
        position_size = self.dna["base_position_size"] * (1 / (normalized_atr * self.dna["volatility_multiplier"]))
        
        atr_limits = config.get("trading.volatility_gene.parameters.atr_limits", {})
        min_size = atr_limits.get("min_size", 1.0)
        max_size = atr_limits.get("max_size", 20.0)
        
        return np.clip(position_size, min_size, max_size)

    def generate_signals(self, market_data: List[MarketData]) -> List[Signal]:
        signals = super().generate_signals(market_data)
        
        if signals and signals[0].type in [SignalType.LONG, SignalType.SHORT]:
            prices = np.array([d.close for d in market_data])
            highs = np.array([d.high for d in market_data])
            lows = np.array([d.low for d in market_data])
            
            self.dna["position_size_pct"] = self.calculate_position_size(prices, highs, lows)
            
        return signals
