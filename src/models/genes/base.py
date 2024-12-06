# src/models/genes/base.py
import numpy as np
from typing import Dict, Optional
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class Operator(Enum):
    GREATER = ">"
    LESS = "<" 
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"

class TradingGene:
    def __init__(self, random_init=True):
        self.fitness_score = None
        
        if random_init:
            self.randomize_dna()
        else:
            self.load_default_dna()

    def randomize_dna(self):
        """Initialize random DNA with valid periods"""
        valid_periods = list(range(5, 200, 5))  # 5, 10, 15, ..., 195
        
        self.dna = {
            "entry_indicator1": np.random.choice(["SMA", "RSI", "BB_UPPER"]),
            "entry_indicator1_params": {
                "timeperiod": np.random.choice(valid_periods)
            },
            "entry_indicator2": np.random.choice(["SMA", "RSI", "BB_LOWER"]),
            "entry_indicator2_params": {
                "timeperiod": np.random.choice(valid_periods)
            },
            "entry_operator": np.random.choice([op.value for op in Operator]),
            "stop_loss_pct": np.random.uniform(0.5, 5.0),
            "take_profit_pct": np.random.uniform(1.0, 10.0)
        }

    def load_default_dna(self):
        """Load default DNA configuration using valid periods"""
        self.dna = {
            "entry_indicator1": "SMA",
            "entry_indicator1_params": {"timeperiod": 20},
            "entry_indicator2": "SMA", 
            "entry_indicator2_params": {"timeperiod": 50},
            "entry_operator": "cross_above",
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0
        }

    def mutate(self, mutation_rate: float = 0.1):
        """Mutate DNA with valid periods only"""
        valid_periods = list(range(5, 200, 5))
        for key in self.dna:
            if np.random.random() < mutation_rate:
                if key == "entry_operator":
                    self.dna[key] = np.random.choice([op.value for op in Operator])
                elif key.endswith("_indicator1") or key.endswith("_indicator2"):
                    self.dna[key] = np.random.choice(["SMA", "RSI", "BB_UPPER", "BB_LOWER"])
                elif key.endswith("_params"):
                    self.dna[key] = {"timeperiod": np.random.choice(valid_periods)}
                elif key.endswith("_pct"):
                    if "stop_loss" in key:
                        self.dna[key] = np.random.uniform(0.5, 5.0)
                    else:
                        self.dna[key] = np.random.uniform(1.0, 10.0)

    def generate_entry_conditions(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        try:
            ind1_key = f"{self.dna['entry_indicator1']}_{self.dna['entry_indicator1_params']['timeperiod']}"
            ind2_key = f"{self.dna['entry_indicator2']}_{self.dna['entry_indicator2_params']['timeperiod']}"
            
            if ind1_key not in data or ind2_key not in data:
                return np.zeros(len(data[list(data.keys())[0]]), dtype=bool)

            values1 = data[ind1_key]
            values2 = data[ind2_key]
            results = np.zeros_like(values1, dtype=bool)

            if self.dna["entry_operator"] == Operator.GREATER.value:
                results[1:] = values1[1:] > values2[1:]
            elif self.dna["entry_operator"] == Operator.LESS.value:
                results[1:] = values1[1:] < values2[1:]
            elif self.dna["entry_operator"] == Operator.CROSS_ABOVE.value:
                results[1:] = (values1[:-1] <= values2[:-1]) & (values1[1:] > values2[1:])
            elif self.dna["entry_operator"] == Operator.CROSS_BELOW.value:
                results[1:] = (values1[:-1] >= values2[:-1]) & (values1[1:] < values2[1:])
                
            return results
            
        except Exception as e:
            logger.error(f"Error in generate_entry_conditions: {e}")
            return np.zeros(len(data[list(data.keys())[0]]), dtype=bool)

    def crossover(self, other: 'TradingGene') -> 'TradingGene':
        """Create child with mixed DNA"""
        child = TradingGene(random_init=False)
        for key in self.dna:
            child.dna[key] = self.dna[key] if np.random.random() < 0.5 else other.dna[key]
        return child