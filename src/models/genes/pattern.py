# src/models/genes/pattern.py
import numpy as np
import talib
import logging
from typing import List, Dict, Optional, Tuple, Callable

from .base import TradingGene, GeneType
from ..common import Signal, SignalType, MarketData
from ...utils.config import config

logger = logging.getLogger(__name__)

class PatternRecognitionGene(TradingGene):
    def __init__(self, random_init=True):
        super().__init__(random_init=False)
        self.gene_type = GeneType.PATTERN.value
        
        # Mappa dei pattern supportati
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
        
        # Carica e valida i pattern abilitati dal config
        self.available_patterns = self._initialize_patterns()
        
        # Carica parametri dal config
        params = config.get("trading.pattern_gene.parameters", {})
        
        if random_init:
            self.initialize_pattern_dna(params)
        else:
            self._initialize_default_dna(params)

    def _initialize_patterns(self) -> Dict[str, Tuple[Callable, int]]:
        """Inizializza i pattern disponibili dal config con validazione"""
        available_patterns = {}
        enabled_patterns = config.get("trading.pattern_gene.patterns", [])
        
        if not enabled_patterns:
            logger.warning("Nessun pattern abilitato nel config")
            return available_patterns
            
        for pattern in enabled_patterns:
            if pattern not in self.pattern_name_map:
                logger.warning(f"Pattern {pattern} non supportato")
                continue
                
            talib_name = self.pattern_name_map[pattern]
            if not hasattr(talib, talib_name):
                logger.warning(f"Funzione TA-Lib {talib_name} non trovata")
                continue
                
            # Determina il numero di candele richieste per il pattern
            candles_required = 2 if pattern in ["ENGULFING", "EVENINGSTAR", "MORNINGSTAR"] else 1
            available_patterns[pattern] = (getattr(talib, talib_name), candles_required)
            logger.info(f"Pattern {pattern} inizializzato con successo")
            
        return available_patterns

    def _initialize_default_dna(self, params: dict):
        """Inizializza DNA con valori di default dal config"""
        pattern_window = params.get("pattern_window", {})
        confirmation = params.get("confirmation_periods", {})
        required = params.get("required_patterns", {})
        
        self.dna.update({
            "required_patterns": required.get("default", 2),
            "pattern_window": pattern_window.get("default", 3),
            "confirmation_periods": confirmation.get("default", 1)
        })
        
        logger.info(f"DNA inizializzato con valori di default: {self.dna}")

    def initialize_pattern_dna(self, params: dict):
        """Inizializza DNA con valori casuali entro i limiti del config"""
        try:
            # Validazione dei parametri pattern window
            pattern_window = params.get("pattern_window", {})
            min_window = pattern_window.get("min", 2)
            max_window = pattern_window.get("max", 7)
            if min_window >= max_window:
                logger.warning("pattern_window.min >= max, usando valori di default")
                min_window, max_window = 2, 7
                
            # Validazione confirmation periods
            conf_periods = params.get("confirmation_periods", {})
            min_conf = conf_periods.get("min", 1)
            max_conf = conf_periods.get("max", 4)
            if min_conf >= max_conf:
                logger.warning("confirmation_periods.min >= max, usando valori di default")
                min_conf, max_conf = 1, 4
                
            # Validazione required patterns
            req_patterns = params.get("required_patterns", {})
            min_req = req_patterns.get("min", 1)
            max_req = req_patterns.get("max", 4)
            if min_req >= max_req:
                logger.warning("required_patterns.min >= max, usando valori di default")
                min_req, max_req = 1, 4
            
            self.dna.update({
                "required_patterns": np.random.randint(min_req, max_req + 1),
                "pattern_window": np.random.randint(min_window, max_window + 1),
                "confirmation_periods": np.random.randint(min_conf, max_conf + 1)
            })
            
            logger.info(f"DNA inizializzato con valori casuali: {self.dna}")
            
        except Exception as e:
            logger.error(f"Errore nell'inizializzazione DNA: {e}")
            # Fallback su valori di default
            self._initialize_default_dna(params)

    def detect_patterns(self, market_data: List[MarketData]) -> Dict[str, int]:
        """Rileva pattern di candele nei dati di mercato"""
        if len(market_data) < max(self.dna["pattern_window"], 10):
            logger.warning("Dati di mercato insufficienti per il rilevamento pattern")
            return {}
            
        try:
            opens = np.array([d.open for d in market_data])
            highs = np.array([d.high for d in market_data])
            lows = np.array([d.low for d in market_data])
            closes = np.array([d.close for d in market_data])
            
            patterns = {}
            window_size = self.dna["pattern_window"]
            
            for name, (func, req_candles) in self.available_patterns.items():
                if len(market_data) < req_candles + window_size:
                    continue
                    
                result = func(opens[-window_size:], 
                            highs[-window_size:], 
                            lows[-window_size:], 
                            closes[-window_size:])
                            
                if not np.isnan(result[-1]):
                    patterns[name] = result[-1]
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Errore nel rilevamento pattern: {e}")
            return {}

    def generate_signals(self, market_data: List[MarketData]) -> List[Signal]:
        """Genera segnali basati sui pattern rilevati"""
        signals = super().generate_signals(market_data)
        
        if not signals:
            return []
            
        try:
            if signals[0].type in [SignalType.LONG, SignalType.SHORT]:
                patterns = self.detect_patterns(market_data)
                
                if not patterns:
                    return []
                    
                bullish_patterns = sum(1 for v in patterns.values() if v > 0)
                bearish_patterns = sum(1 for v in patterns.values() if v < 0)
                
                signal_type = signals[0].type
                confirmation_required = self.dna["confirmation_periods"]
                
                # Verifica conferma pattern
                if signal_type == SignalType.LONG:
                    if bullish_patterns < self.dna["required_patterns"]:
                        logger.debug(f"Pattern bullish insufficienti: {bullish_patterns}")
                        return []
                    # Verifica periodi di conferma
                    for i in range(confirmation_required):
                        if i + 1 >= len(market_data) or market_data[-i-2].close > market_data[-i-1].close:
                            return []
                            
                elif signal_type == SignalType.SHORT:
                    if bearish_patterns < self.dna["required_patterns"]:
                        logger.debug(f"Pattern bearish insufficienti: {bearish_patterns}")
                        return []
                    # Verifica periodi di conferma
                    for i in range(confirmation_required):
                        if i + 1 >= len(market_data) or market_data[-i-2].close < market_data[-i-1].close:
                            return []
                            
                return signals
                
        except Exception as e:
            logger.error(f"Errore nella generazione segnali: {e}")
            return []
            
        return signals
