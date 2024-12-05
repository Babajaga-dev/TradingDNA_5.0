# src/models/genes/base.py
from enum import Enum
from typing import Dict, List
import numpy as np
from datetime import datetime

from ..common import Signal, SignalType, MarketData
from ...utils.config import config

class Operator(Enum):
    GREATER = ">"
    LESS = "<"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"

class TradingGene:
    def __init__(self, random_init=True):
        self.fitness_score = None
        self.performance_history = {}
        self.mutation_history = []
        
        if random_init:
            self._initialize_random()
        else:
            self._load_from_config()

    def _initialize_random(self):
        self.dna = {
            "entry_indicator1": np.random.choice(list(self.available_indicators.keys())),
            "entry_indicator1_params": self._generate_random_params(),
            "entry_indicator2": np.random.choice(list(self.available_indicators.keys())),
            "entry_indicator2_params": self._generate_random_params(),
            "entry_operator": np.random.choice(list(Operator)),
            "exit_indicator1": np.random.choice(list(self.available_indicators.keys())),
            "exit_indicator1_params": self._generate_random_params(),
            "exit_indicator2": np.random.choice(list(self.available_indicators.keys())),
            "exit_indicator2_params": self._generate_random_params(),
            "exit_operator": np.random.choice(list(Operator)),
            "stop_loss_pct": np.random.uniform(0.5, 5.0),
            "take_profit_pct": np.random.uniform(1.0, 10.0),
            "position_size_pct": np.random.uniform(1.0, 20.0)
        }

    def _load_from_config(self):
        self.dna = {
            "entry_indicator1": config.get("trading.indicators.entry.indicator1.type", "SMA"),
            "entry_indicator1_params": config.get("trading.indicators.entry.indicator1.params", {"timeperiod": 20}),
            "entry_indicator2": config.get("trading.indicators.entry.indicator2.type", "SMA"),
            "entry_indicator2_params": config.get("trading.indicators.entry.indicator2.params", {"timeperiod": 50}),
            "entry_operator": Operator(config.get("trading.indicators.entry.operator", "cross_above")),
            "exit_indicator1": config.get("trading.indicators.exit.indicator1.type", "RSI"),
            "exit_indicator1_params": config.get("trading.indicators.exit.indicator1.params", {"timeperiod": 14}),
            "exit_indicator2": config.get("trading.indicators.exit.indicator2.type", "CLOSE"),
            "exit_indicator2_params": config.get("trading.indicators.exit.indicator2.params", {}),
            "exit_operator": Operator(config.get("trading.indicators.exit.operator", ">")),
            "stop_loss_pct": config.get("trading.position.stop_loss_pct", 2.0),
            "take_profit_pct": config.get("trading.position.take_profit_pct", 4.0),
            "position_size_pct": config.get("trading.position.size_pct", 5.0)
        }

    def _generate_random_params(self):
        """Genera parametri casuali per indicatori"""
        params = {}
        # implementazione in indicators.py
        return params

    def mutate(self, mutation_rate: float = 0.1):
        """Muta il DNA del gene"""
        mutations = []
        for key in self.dna:
            if np.random.random() < mutation_rate:
                mutations.extend(self._mutate_parameter(key))
        
        if mutations:
            self._log_mutations(mutations)
            self.validate_dna()

    def crossover(self, other: 'TradingGene') -> 'TradingGene':
        """Esegue crossover con altro gene"""
        child = self.__class__(random_init=False)
        for key in self.dna:
            child.dna[key] = self.dna[key] if np.random.random() < 0.5 else other.dna[key]
        return child

    def validate_dna(self):
        """Valida e corregge DNA se necessario"""
        # implementazione validation logic
        pass

    def generate_signals(self, market_data: List[MarketData]) -> List[Signal]:
        """Genera segnali di trading basati sul DNA"""
        # implementazione generazione segnali
        pass