# src/models/genes/volatility.py
import numpy as np
import logging
from typing import List, Dict, Optional

from .base import TradingGene, GeneType
from .indicators import calculate_atr
from ..common import Signal, SignalType, MarketData
from ...utils.config import config

logger = logging.getLogger(__name__)

class VolatilityAdaptiveGene(TradingGene):
    def __init__(self, random_init=True):
        super().__init__(random_init=False)
        self.gene_type = GeneType.VOLATILITY.value
        
        # Carica e valida i parametri dal config
        params = self._validate_parameters(config.get("trading.volatility_gene.parameters", {}))
        
        if random_init:
            self._initialize_random_dna(params)
        else:
            self._initialize_default_dna(params)
            
        logger.info(f"VolatilityAdaptiveGene inizializzato con DNA: {self.dna}")

    def _validate_parameters(self, params: Dict) -> Dict:
        """Valida i parametri dal config e imposta valori di default se necessario"""
        validated = {}
        
        # Timeperiod
        timeperiod = params.get("timeperiod", {})
        validated["timeperiod"] = {
            "min": max(2, timeperiod.get("min", 10)),
            "max": max(10, timeperiod.get("max", 50)),
            "default": max(5, timeperiod.get("default", 14))
        }
        
        # Multiplier
        multiplier = params.get("multiplier", {})
        validated["multiplier"] = {
            "min": max(0.1, multiplier.get("min", 0.5)),
            "max": max(1.0, multiplier.get("max", 2.0)),
            "default": max(0.5, multiplier.get("default", 1.0))
        }
        
        # Base position size
        base_size = params.get("base_position_size", {})
        validated["base_position_size"] = {
            "min": max(0.1, base_size.get("min", 1.0)),
            "max": max(1.0, base_size.get("max", 10.0)),
            "default": max(0.5, base_size.get("default", 5.0))
        }
        
        # ATR limits
        atr_limits = params.get("atr_limits", {})
        validated["atr_limits"] = {
            "min_size": max(0.1, atr_limits.get("min_size", 0.5)),
            "max_size": max(1.0, atr_limits.get("max_size", 25.0))
        }
        
        # Verifica che i min siano minori dei max
        for param in ["timeperiod", "multiplier", "base_position_size"]:
            if validated[param]["min"] >= validated[param]["max"]:
                logger.warning(f"Parametro {param}: min >= max, uso valori di default")
                if param == "timeperiod":
                    validated[param].update({"min": 10, "max": 50, "default": 14})
                elif param == "multiplier":
                    validated[param].update({"min": 0.5, "max": 2.0, "default": 1.0})
                else:  # base_position_size
                    validated[param].update({"min": 1.0, "max": 10.0, "default": 5.0})
                    
        # Verifica ATR limits
        if validated["atr_limits"]["min_size"] >= validated["atr_limits"]["max_size"]:
            logger.warning("ATR limits: min_size >= max_size, uso valori di default")
            validated["atr_limits"].update({"min_size": 0.5, "max_size": 25.0})
            
        return validated

    def _initialize_random_dna(self, params: Dict):
        """Inizializza DNA con valori casuali entro i limiti validati"""
        try:
            self.dna.update({
                "volatility_timeperiod": np.random.randint(
                    params["timeperiod"]["min"],
                    params["timeperiod"]["max"] + 1
                ),
                "volatility_multiplier": np.random.uniform(
                    params["multiplier"]["min"],
                    params["multiplier"]["max"]
                ),
                "base_position_size": np.random.uniform(
                    params["base_position_size"]["min"],
                    params["base_position_size"]["max"]
                )
            })
        except Exception as e:
            logger.error(f"Errore nell'inizializzazione random DNA: {e}")
            self._initialize_default_dna(params)

    def _initialize_default_dna(self, params: Dict):
        """Inizializza DNA con valori di default"""
        self.dna.update({
            "volatility_timeperiod": params["timeperiod"]["default"],
            "volatility_multiplier": params["multiplier"]["default"],
            "base_position_size": params["base_position_size"]["default"]
        })

    def calculate_position_size(self, prices: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
        """Calcola la dimensione della posizione basata sulla volatilità"""
        try:
            if len(prices) < self.dna["volatility_timeperiod"]:
                logger.warning("Dati insufficienti per il calcolo ATR")
                return self.dna["base_position_size"]
                
            # Calcola ATR
            atr = calculate_atr(high, low, prices, self.dna["volatility_timeperiod"])
            if np.isnan(atr[-1]):
                logger.warning("ATR ha prodotto un valore NaN")
                return self.dna["base_position_size"]
                
            # Calcola prezzo medio per normalizzazione
            recent_prices = prices[-self.dna["volatility_timeperiod"]:]
            if len(recent_prices) == 0:
                logger.warning("Nessun prezzo recente disponibile")
                return self.dna["base_position_size"]
                
            avg_price = np.mean(recent_prices)
            if avg_price == 0:
                logger.warning("Prezzo medio è zero")
                return self.dna["base_position_size"]
                
            # Calcola ATR normalizzato
            normalized_atr = atr[-1] / avg_price
            
            # Calcola dimensione posizione
            position_size = self.dna["base_position_size"] * (1 / (normalized_atr * self.dna["volatility_multiplier"]))
            
            # Applica limiti ATR
            params = config.get("trading.volatility_gene.parameters", {})
            atr_limits = params.get("atr_limits", {})
            min_size = atr_limits.get("min_size", 0.5)
            max_size = atr_limits.get("max_size", 25.0)
            
            clipped_size = np.clip(position_size, min_size, max_size)
            
            logger.debug(f"Dimensione posizione calcolata: {position_size}, "
                        f"dopo clip: {clipped_size} (min: {min_size}, max: {max_size})")
            
            return clipped_size
            
        except Exception as e:
            logger.error(f"Errore nel calcolo della dimensione posizione: {e}")
            return self.dna["base_position_size"]

    def generate_signals(self, market_data: List[MarketData]) -> List[Signal]:
        """Genera segnali con dimensione posizione adattiva"""
        signals = super().generate_signals(market_data)
        
        if signals and signals[0].type in [SignalType.LONG, SignalType.SHORT]:
            try:
                prices = np.array([d.close for d in market_data])
                highs = np.array([d.high for d in market_data])
                lows = np.array([d.low for d in market_data])
                
                position_size = self.calculate_position_size(prices, highs, lows)
                self.dna["position_size_pct"] = position_size
                
                logger.debug(f"Segnale generato con position_size_pct: {position_size}")
                
            except Exception as e:
                logger.error(f"Errore nella generazione del segnale: {e}")
                self.dna["position_size_pct"] = self.dna["base_position_size"]
                
        return signals
