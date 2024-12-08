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
    # Indicatori che richiedono timeperiod
    TIMEPERIOD_INDICATORS = ["SMA", "EMA", "RSI", "BB_UPPER", "BB_LOWER"]
    # Tutti gli indicatori validi
    VALID_INDICATORS = TIMEPERIOD_INDICATORS + ["CLOSE"]
    # Periodi validi per gli indicatori
    VALID_PERIODS = [5, 8, 13, 21, 34, 55, 89]  # Periodi di Fibonacci

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
            # Usa un periodo di default se manca
            return f"{indicator}_21"
        return f"{indicator}_{params['timeperiod']}"

    def _get_indicator_params(self, indicator: str) -> Dict:
        """Restituisce i parametri corretti per un indicatore"""
        if indicator not in self.TIMEPERIOD_INDICATORS:
            return {}
        return {"timeperiod": int(np.random.choice(self.VALID_PERIODS))}

    def randomize_dna(self):
        """Initialize random DNA with valid periods"""
        try:
            # Scegli indicatori casuali
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
        """Load default DNA configuration"""
        self.dna = {
            "entry_indicator1": "SMA",
            "entry_indicator1_params": {"timeperiod": 21},
            "entry_indicator2": "SMA", 
            "entry_indicator2_params": {"timeperiod": 55},
            "entry_operator": "cross_above",
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0
        }

    def generate_entry_conditions(self, data: Dict[str, np.ndarray]) -> np.ndarray:
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

    def mutate(self, mutation_rate: float = 0.1):
        """Mutate DNA with valid periods only"""
        try:
            for key in self.dna:
                if np.random.random() < mutation_rate:
                    if key == "entry_operator":
                        self.dna[key] = np.random.choice([op.value for op in Operator])
                    elif key.endswith("_indicator1") or key.endswith("_indicator2"):
                        self.dna[key] = np.random.choice(self.VALID_INDICATORS)
                        # Aggiorna i parametri in base al nuovo indicatore
                        param_key = f"{key}_params"
                        self.dna[param_key] = self._get_indicator_params(self.dna[key])
                    elif key.endswith("_params"):
                        base_key = key.replace("_params", "")
                        if self.dna[base_key] in self.TIMEPERIOD_INDICATORS:
                            self.dna[key] = {"timeperiod": int(np.random.choice(self.VALID_PERIODS))}
                    elif key.endswith("_pct"):
                        if "stop_loss" in key:
                            self.dna[key] = float(np.random.uniform(0.5, 5.0))
                        else:
                            self.dna[key] = float(np.random.uniform(1.0, 10.0))
                            
            logger.debug(f"Mutated DNA: {self.dna}")
            
        except Exception as e:
            logger.error(f"Error in mutate: {str(e)}")

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