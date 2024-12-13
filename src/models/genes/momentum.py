# src/models/genes/momentum.py
import numpy as np
import talib
import logging
from typing import List, Dict, Optional

from .base import TradingGene, GeneType
from .indicators import calculate_stoch
from ..common import Signal, SignalType, MarketData
from ...utils.config import config

logger = logging.getLogger(__name__)

class MomentumGene(TradingGene):
    def __init__(self, random_init=True):
        super().__init__(random_init=False)
        self.gene_type = GeneType.MOMENTUM.value
        
        # Carica e valida i parametri dal config
        params = self._validate_parameters(config.get("trading.momentum_gene.parameters", {}))
        
        if random_init:
            self._initialize_random_dna(params)
        else:
            self._initialize_default_dna(params)
            
        logger.info(f"MomentumGene inizializzato con DNA: {self.dna}")

    def _validate_parameters(self, params: Dict) -> Dict:
        """Valida i parametri dal config e imposta valori di default se necessario"""
        validated = {}
        
        # Momentum threshold
        momentum = params.get("momentum_threshold", {})
        validated["momentum_threshold"] = {
            "min": max(0, min(100, momentum.get("min", 60))),
            "max": max(0, min(100, momentum.get("max", 80))),
            "default": max(0, min(100, momentum.get("default", 70)))
        }
        
        # Trend strength
        trend = params.get("trend_strength", {})
        validated["trend_strength"] = {
            "min": max(0, min(100, trend.get("min", 20))),
            "max": max(0, min(100, trend.get("max", 30))),
            "default": max(0, min(100, trend.get("default", 25)))
        }
        
        # Overbought level
        overbought = params.get("overbought_level", {})
        validated["overbought_level"] = {
            "min": max(50, min(100, overbought.get("min", 75))),
            "max": max(50, min(100, overbought.get("max", 85))),
            "default": max(50, min(100, overbought.get("default", 80)))
        }
        
        # Oversold level
        oversold = params.get("oversold_level", {})
        validated["oversold_level"] = {
            "min": max(0, min(50, oversold.get("min", 15))),
            "max": max(0, min(50, oversold.get("max", 25))),
            "default": max(0, min(50, oversold.get("default", 20)))
        }
        
        # Verifica che i min siano minori dei max
        for param in validated:
            if validated[param]["min"] >= validated[param]["max"]:
                logger.warning(f"Parametro {param}: min >= max, uso valori di default")
                if param == "oversold_level":
                    validated[param]["min"] = 15
                    validated[param]["max"] = 25
                    validated[param]["default"] = 20
                elif param == "overbought_level":
                    validated[param]["min"] = 75
                    validated[param]["max"] = 85
                    validated[param]["default"] = 80
                else:
                    validated[param]["min"] = 20
                    validated[param]["max"] = 80
                    validated[param]["default"] = 50
                    
        return validated

    def _initialize_random_dna(self, params: Dict):
        """Inizializza DNA con valori casuali entro i limiti validati"""
        try:
            self.dna.update({
                "momentum_threshold": np.random.randint(
                    params["momentum_threshold"]["min"],
                    params["momentum_threshold"]["max"] + 1
                ),
                "trend_strength_threshold": np.random.randint(
                    params["trend_strength"]["min"],
                    params["trend_strength"]["max"] + 1
                ),
                "overbought_level": np.random.randint(
                    params["overbought_level"]["min"],
                    params["overbought_level"]["max"] + 1
                ),
                "oversold_level": np.random.randint(
                    params["oversold_level"]["min"],
                    params["oversold_level"]["max"] + 1
                )
            })
        except Exception as e:
            logger.error(f"Errore nell'inizializzazione random DNA: {e}")
            self._initialize_default_dna(params)

    def _initialize_default_dna(self, params: Dict):
        """Inizializza DNA con valori di default"""
        self.dna.update({
            "momentum_threshold": params["momentum_threshold"]["default"],
            "trend_strength_threshold": params["trend_strength"]["default"],
            "overbought_level": params["overbought_level"]["default"],
            "oversold_level": params["oversold_level"]["default"]
        })

    def _calculate_indicators(self, market_data: List[MarketData]) -> Optional[Dict]:
        """Calcola gli indicatori necessari per il momentum"""
        try:
            if len(market_data) < 14:  # Minimo periodo necessario
                logger.warning("Dati di mercato insufficienti per calcolare gli indicatori")
                return None
                
            prices = np.array([d.close for d in market_data])
            highs = np.array([d.high for d in market_data])
            lows = np.array([d.low for d in market_data])
            
            # Carica parametri indicatori
            rsi_params = config.get("trading.momentum_gene.parameters.rsi", {})
            stoch_params = config.get("trading.momentum_gene.parameters.stochastic", {})
            adx_params = config.get("trading.momentum_gene.parameters.adx", {})
            
            # Calcola indicatori
            rsi = talib.RSI(prices, timeperiod=rsi_params.get("timeperiod", 14))
            stoch = calculate_stoch(
                highs, lows, prices,
                fastk_period=stoch_params.get("fastk_period", 14),
                slowk_period=stoch_params.get("slowk_period", 3),
                slowd_period=stoch_params.get("slowd_period", 3)
            )
            adx = talib.ADX(highs, lows, prices, timeperiod=adx_params.get("timeperiod", 14))
            
            if np.isnan(rsi[-1]) or np.isnan(stoch[-1]) or np.isnan(adx[-1]):
                logger.warning("Uno o più indicatori hanno prodotto valori NaN")
                return None
                
            return {
                "rsi": rsi[-1],
                "stoch": stoch[-1],
                "adx": adx[-1]
            }
            
        except Exception as e:
            logger.error(f"Errore nel calcolo degli indicatori: {e}")
            return None

    def check_momentum_conditions(self, market_data: List[MarketData]) -> bool:
        """Verifica le condizioni di momentum"""
        indicators = self._calculate_indicators(market_data)
        if not indicators:
            return False
            
        try:
            # Verifica trend forte
            strong_trend = indicators["adx"] > self.dna["trend_strength_threshold"]
            
            # Verifica condizioni di ipercomprato/ipervenduto
            overbought = (indicators["rsi"] > self.dna["overbought_level"] and 
                         indicators["stoch"] > self.dna["overbought_level"])
            oversold = (indicators["rsi"] < self.dna["oversold_level"] and 
                       indicators["stoch"] < self.dna["oversold_level"])
            
            momentum_condition = strong_trend and (overbought or oversold)
            
            logger.debug(f"Condizioni momentum: strong_trend={strong_trend}, "
                        f"overbought={overbought}, oversold={oversold}")
            
            return momentum_condition
            
        except Exception as e:
            logger.error(f"Errore nella verifica delle condizioni momentum: {e}")
            return False

    def generate_signals(self, market_data: List[MarketData]) -> List[Signal]:
        """Genera segnali basati sulle condizioni di momentum"""
        signals = super().generate_signals(market_data)
        
        if signals and signals[0].type in [SignalType.LONG, SignalType.SHORT]:
            if not self.check_momentum_conditions(market_data):
                logger.debug("Condizioni momentum non soddisfatte")
                return []
                
        return signals
