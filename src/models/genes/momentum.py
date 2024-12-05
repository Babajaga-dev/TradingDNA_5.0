# src/models/genes/momentum.py
import numpy as np
import talib
from typing import List

from .base import TradingGene
from .indicators import calculate_stoch
from ..common import Signal, SignalType, MarketData
from ...utils.config import config

class MomentumGene(TradingGene):
    def __init__(self, random_init=True):
        super().__init__(random_init=False)
        
        params = config.get("trading.momentum_gene.parameters", {})
        
        if random_init:
            self.dna.update({
                "momentum_threshold": np.random.randint(
                    params.get("momentum_threshold", {}).get("min", 60),
                    params.get("momentum_threshold", {}).get("max", 80)
                ),
                "trend_strength_threshold": np.random.randint(
                    params.get("trend_strength", {}).get("min", 20),
                    params.get("trend_strength", {}).get("max", 30)
                ),
                "overbought_level": np.random.randint(
                    params.get("overbought_level", {}).get("min", 75),
                    params.get("overbought_level", {}).get("max", 85)
                ),
                "oversold_level": np.random.randint(
                    params.get("oversold_level", {}).get("min", 15),
                    params.get("oversold_level", {}).get("max", 25)
                )
            })
        else:
            self.dna.update({
                "momentum_threshold": params.get("momentum_threshold", {}).get("default", 70),
                "trend_strength_threshold": params.get("trend_strength", {}).get("default", 25),
                "overbought_level": params.get("overbought_level", {}).get("default", 80),
                "oversold_level": params.get("oversold_level", {}).get("default", 20)
            })

    def check_momentum_conditions(self, market_data: List[MarketData]) -> bool:
        prices = np.array([d.close for d in market_data])
        highs = np.array([d.high for d in market_data])
        lows = np.array([d.low for d in market_data])
        
        rsi_params = config.get("trading.momentum_gene.parameters.rsi", {})
        stoch_params = config.get("trading.momentum_gene.parameters.stochastic", {})
        adx_params = config.get("trading.momentum_gene.parameters.adx", {})
        
        rsi = talib.RSI(prices, timeperiod=rsi_params.get("timeperiod", 14))
        stoch = calculate_stoch(highs, lows, prices,
                              fastk_period=stoch_params.get("fastk_period", 14),
                              slowk_period=stoch_params.get("slowk_period", 3),
                              slowd_period=stoch_params.get("slowd_period", 3))
        adx = talib.ADX(highs, lows, prices, timeperiod=adx_params.get("timeperiod", 14))
        
        if np.isnan(rsi[-1]) or np.isnan(stoch[-1]) or np.isnan(adx[-1]):
            return False
            
        strong_trend = adx[-1] > self.dna["trend_strength_threshold"]
        overbought = (rsi[-1] > self.dna["overbought_level"] and 
                     stoch[-1] > self.dna["overbought_level"])
        oversold = (rsi[-1] < self.dna["oversold_level"] and 
                   stoch[-1] < self.dna["oversold_level"])
        
        return strong_trend and (overbought or oversold)

    def generate_signals(self, market_data: List[MarketData]) -> List[Signal]:
        signals = super().generate_signals(market_data)
        
        if signals and signals[0].type in [SignalType.LONG, SignalType.SHORT]:
            if not self.check_momentum_conditions(market_data):
                return []
                
        return signals