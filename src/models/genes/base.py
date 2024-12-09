# src/models/genes/base.py
import traceback
import numpy as np
from typing import Dict, Optional, Tuple
import logging
import random
from enum import Enum

logger = logging.getLogger(__name__)

class Operator(Enum):
    GREATER = ">"
    LESS = "<" 
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"

class TradingGene:
    # Indicatori che richiedono timeperiod
    TIMEPERIOD_INDICATORS = ["SMA", "EMA", "RSI", "BB_UPPER", "BB_LOWER"]
    
    # Tutti gli indicatori validi
    VALID_INDICATORS = TIMEPERIOD_INDICATORS + ["CLOSE"]
    
    # Periodi validi estesi per gli indicatori
    VALID_PERIODS = [
        # Fibonacci sequence
        3, 5, 8, 13, 21, 34, 55, 89, 144,
        # Standard trading periods
        10, 20, 50, 100, 200,
        # Intraday periods
        15, 30, 60,
    ]
    
    # Raggruppa i periodi per tipo di analisi
    PERIOD_GROUPS = {
        'short_term': [3, 5, 8, 10, 13, 15],
        'medium_term': [20, 21, 30, 34, 50, 55],
        'long_term': [60, 89, 100, 144, 200]
    }

    def __init__(self, random_init=True):
        self.fitness_score = None
        self.dna = {}
        
        if random_init:
            self.randomize_dna()
        else:
            self.load_default_dna()

    def _get_indicator_key(self, indicator: str, params: Dict) -> str:
        """Genera la chiave dell'indicatore nel formato corretto"""
        if indicator not in self.TIMEPERIOD_INDICATORS:
            return indicator
        if "timeperiod" not in params:
            logger.error(f"Missing timeperiod for indicator {indicator}")
            return f"{indicator}_21"  # Default period
        return f"{indicator}_{params['timeperiod']}"

    def _get_indicator_params(self, indicator: str) -> Dict:
        """Restituisce i parametri corretti per un indicatore con selezione intelligente dei periodi"""
        if indicator not in self.TIMEPERIOD_INDICATORS:
            return {}

        # Intelligente selezione dei periodi basata sul contesto
        if np.random.random() < 0.7:  # 70% delle volte usa periodi dello stesso gruppo
            group = np.random.choice(list(self.PERIOD_GROUPS.keys()))
            period = np.random.choice(self.PERIOD_GROUPS[group])
        else:  # 30% delle volte sceglie completamente random
            period = np.random.choice(self.VALID_PERIODS)

        return {"timeperiod": int(period)}

    def generate_entry_conditions(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Genera le condizioni di ingresso basate sul DNA"""
        try:
            # Verifica la presenza di tutti i parametri necessari
            required_keys = ["entry_indicator1", "entry_indicator2", "entry_operator"]
            if not all(k in self.dna for k in required_keys):
                missing_keys = [k for k in required_keys if k not in self.dna]
                logger.error(f"Missing DNA keys: {missing_keys}")
                return np.zeros(len(next(iter(data.values()))), dtype=bool)

            # Ottieni i parametri e assicurati che siano validi
            ind1 = self.dna["entry_indicator1"]
            ind2 = self.dna["entry_indicator2"]
            params1 = self.dna.get("entry_indicator1_params", {})
            params2 = self.dna.get("entry_indicator2_params", {})

            # Se mancano i parametri per indicatori che li richiedono, usa valori di default
            if ind1 in self.TIMEPERIOD_INDICATORS and "timeperiod" not in params1:
                params1 = {"timeperiod": 21}
                self.dna["entry_indicator1_params"] = params1
            if ind2 in self.TIMEPERIOD_INDICATORS and "timeperiod" not in params2:
                params2 = {"timeperiod": 55}
                self.dna["entry_indicator2_params"] = params2
            
            # Genera le chiavi degli indicatori
            ind1_key = self._get_indicator_key(ind1, params1)
            ind2_key = self._get_indicator_key(ind2, params2)

            # Debug log
            logger.debug(f"Looking for indicators: {ind1_key} and {ind2_key}")
            logger.debug(f"Available indicators: {list(data.keys())}")
            
            # Verifica presenza indicatori
            if ind1_key not in data or ind2_key not in data:
                logger.error(f"Indicators not found. Looking for {ind1_key} and {ind2_key}")
                logger.error(f"Available indicators: {list(data.keys())}")
                return np.zeros(len(next(iter(data.values()))), dtype=bool)

            # Ottieni i valori
            values1 = data[ind1_key]
            values2 = data[ind2_key]
            results = np.zeros_like(values1, dtype=bool)

            # Genera i segnali
            if self.dna["entry_operator"] == Operator.GREATER.value:
                results[1:] = values1[1:] > values2[1:]
            elif self.dna["entry_operator"] == Operator.LESS.value:
                results[1:] = values1[1:] < values2[1:]
            elif self.dna["entry_operator"] == Operator.CROSS_ABOVE.value:
                results[1:] = (values1[:-1] <= values2[:-1]) & (values1[1:] > values2[1:])
            elif self.dna["entry_operator"] == Operator.CROSS_BELOW.value:
                results[1:] = (values1[:-1] >= values2[:-1]) & (values1[1:] < values2[1:])

            return results
            
        except KeyError as e:
            logger.error(f"KeyError in generate_entry_conditions: {str(e)}")
            logger.error(f"DNA: {self.dna}")
            return np.zeros(len(next(iter(data.values()))), dtype=bool)
        except Exception as e:
            logger.error(f"Error in generate_entry_conditions: {str(e)}")
            return np.zeros(len(next(iter(data.values()))), dtype=bool)

    def _get_common_indicator_pair(self) -> Tuple[str, str]:
        """Restituisce coppie comuni di indicatori"""
        common_pairs = [
            ("SMA", "SMA"),     # Moving Average Crossover
            ("EMA", "EMA"),     # Exponential MA Crossover
            ("RSI", "CLOSE"),   # RSI with price
            ("BB_UPPER", "CLOSE"), # Bollinger Band strategy
            ("BB_LOWER", "CLOSE"),
            ("SMA", "EMA"),     # Mixed MA strategy
        ]
        return random.choice(common_pairs)

    def randomize_dna(self):
        """Initialize random DNA with strategically selected periods"""
        try:
            # Scelta strategica degli indicatori
            if np.random.random() < 0.6:  # 60% delle volte usa combinazioni comuni
                ind1, ind2 = self._get_common_indicator_pair()
            else:  # 40% delle volte sceglie completamente random
                ind1 = np.random.choice(self.VALID_INDICATORS)
                ind2 = np.random.choice(self.VALID_INDICATORS)
            
            self.dna = {
                "entry_indicator1": ind1,
                "entry_indicator1_params": self._get_indicator_params(ind1),
                "entry_indicator2": ind2,
                "entry_indicator2_params": self._get_indicator_params(ind2),
                "entry_operator": np.random.choice([op.value for op in Operator]),
                "stop_loss_pct": float(np.random.uniform(0.5, 5.0)),
                "take_profit_pct": float(np.random.uniform(1.0, 10.0))
            }
            
            logger.debug(f"Randomized DNA: {self.dna}")
            
        except Exception as e:
            logger.error(f"Error in randomize_dna: {str(e)}")
            self.load_default_dna()

    def load_default_dna(self):
        """Load default DNA configuration with medium-term periods"""
        self.dna = {
            "entry_indicator1": "SMA",
            "entry_indicator1_params": {"timeperiod": 21},
            "entry_indicator2": "SMA", 
            "entry_indicator2_params": {"timeperiod": 55},
            "entry_operator": "cross_above",
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0
        }

    def mutate(self, mutation_rate: float = 0.1):
        """Mutazione migliorata del DNA con controlli di validità"""
        try:
            for key in self.dna:
                if np.random.random() < mutation_rate:
                    if key == "entry_operator":
                        # Garantisce la scelta di un operatore diverso
                        current_op = self.dna[key]
                        available_ops = [op.value for op in Operator if op.value != current_op]
                        self.dna[key] = np.random.choice(available_ops)
                        
                    elif key.endswith("_indicator1") or key.endswith("_indicator2"):
                        # Garantisce la scelta di un indicatore diverso
                        current_ind = self.dna[key]
                        available_inds = [ind for ind in self.VALID_INDICATORS if ind != current_ind]
                        if available_inds:
                            self.dna[key] = np.random.choice(available_inds)
                            # Aggiorna i parametri in base al nuovo indicatore
                            param_key = f"{key}_params"
                            self.dna[param_key] = self._get_indicator_params(self.dna[key])
                            
                    elif key.endswith("_params"):
                        base_key = key.replace("_params", "")
                        if self.dna[base_key] in self.TIMEPERIOD_INDICATORS:
                            # Mutazione intelligente dei parametri
                            current_period = int(self.dna[key].get("timeperiod", 21))
                            available_periods = np.array([int(p) for p in self.VALID_PERIODS if int(p) != current_period])
                            
                            if len(available_periods) > 0:
                                # Preferisce periodi vicini a quello corrente
                                weights = 1 / (np.abs(available_periods - current_period) + 1)
                                weights = weights / weights.sum()
                                new_period = np.random.choice(available_periods, p=weights)
                                self.dna[key] = {"timeperiod": int(new_period)}
                                
                    elif key.endswith("_pct"):
                        # Mutazione dei parametri percentuali con limiti
                        if "stop_loss" in key:
                            current = float(self.dna[key])
                            # Mutazione gaussiana con limiti
                            delta = np.random.normal(0, 0.5)  # std=0.5%
                            new_value = current + delta
                            self.dna[key] = float(np.clip(new_value, 0.5, 5.0))
                        elif "take_profit" in key:
                            current = float(self.dna[key])
                            delta = np.random.normal(0, 1.0)  # std=1.0%
                            new_value = current + delta
                            self.dna[key] = float(np.clip(new_value, 1.0, 10.0))
                            
                        # Mantiene take_profit > stop_loss
                        if "stop_loss_pct" in self.dna and "take_profit_pct" in self.dna:
                            if self.dna["take_profit_pct"] <= self.dna["stop_loss_pct"]:
                                self.dna["take_profit_pct"] = self.dna["stop_loss_pct"] * 1.5
                    
                    logger.debug(f"Mutated {key}: {self.dna[key]}")
                    
            return self
            
        except Exception as e:
            logger.error(f"Error in mutate: {str(e)}")
            logger.error(traceback.format_exc())
            return self

    def crossover(self, other: 'TradingGene') -> 'TradingGene':
        """Create child with mixed DNA"""
        try:
            child = TradingGene(random_init=False)
            child.dna = {}
            
            for key in self.dna:
                # Gestione speciale per gli indicatori e i loro parametri
                if key.endswith("_indicator1") or key.endswith("_indicator2"):
                    # Sceglie l'indicatore da uno dei genitori
                    indicator = self.dna[key] if np.random.random() < 0.5 else other.dna[key]
                    child.dna[key] = indicator
                    # Imposta i parametri appropriati
                    param_key = f"{key}_params"
                    child.dna[param_key] = self._get_indicator_params(indicator)
                elif not key.endswith("_params"):  # Salta i parametri perché già gestiti sopra
                    # Crossover per gli altri valori
                    child.dna[key] = self.dna[key] if np.random.random() < 0.5 else other.dna[key]
            
            logger.debug(f"Created child DNA: {child.dna}")
            return child
            
        except Exception as e:
            logger.error(f"Error in crossover: {str(e)}")
            return TradingGene(random_init=True)