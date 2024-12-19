import traceback
import numpy as np
import torch
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

class GeneType(Enum):
    BASE = "base_gene"
    MOMENTUM = "momentum_gene"
    VOLATILITY = "volatility_gene"
    PATTERN = "pattern_gene"

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
        from ...utils.config import config
        self.config = config
        self.fitness_score = None
        self.dna = {}
        self.gene_type = GeneType.BASE.value
        
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

    def _to_tensor(self, data):
        """Converte in modo sicuro i dati in torch tensor"""
        try:
            if isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            else:
                return torch.tensor(data)
        except Exception as e:
            logger.error(f"Error converting to tensor: {str(e)}")
            if isinstance(data, np.ndarray):
                shape = data.shape
                dtype = data.dtype
                logger.error(f"Array info - Shape: {shape}, Dtype: {dtype}")
            return torch.tensor([])

    def generate_entry_conditions(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Genera le condizioni di ingresso basate sul DNA"""
        try:
            # Verifica la presenza di tutti i parametri necessari
            required_keys = ["entry_indicator1", "entry_indicator2", "entry_operator"]
            if not all(k in self.dna for k in required_keys):
                missing_keys = [k for k in required_keys if k not in self.dna]
                logger.error(f"Missing DNA keys: {missing_keys}")
                return torch.zeros(len(next(iter(data.values()))), dtype=torch.bool)

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
                return torch.zeros(len(next(iter(data.values()))), dtype=torch.bool)

            # Ottieni la dimensione corretta dai dati di input
            expected_size = len(next(iter(data.values())))
            
            # Ottieni i valori come tensori e assicurati che siano sullo stesso device
            values1 = self._to_tensor(data[ind1_key])
            values2 = self._to_tensor(data[ind2_key])
            
            if len(values1) == 0 or len(values2) == 0:
                logger.error("Error converting indicator values to tensors")
                return torch.zeros(expected_size, dtype=torch.bool)

            # Determina il device corretto
            target_device = values1.device
            values2 = values2.to(target_device)
            
            # Crea il tensore dei risultati con la dimensione corretta
            results = torch.zeros(expected_size, dtype=torch.bool, device=target_device)

            # Genera i segnali con controllo densità
            if self.dna["entry_operator"] == Operator.GREATER.value:
                # Crea segnali con la dimensione corretta
                base_signals = torch.zeros(expected_size, dtype=torch.bool, device=target_device)
                # Assicurati che gli slice abbiano la stessa dimensione
                n = min(len(values1), len(values2), expected_size)
                base_signals[1:n] = values1[1:n] > values2[1:n]
                
                results = base_signals
            elif self.dna["entry_operator"] == Operator.LESS.value:
                # Crea segnali con la dimensione corretta
                base_signals = torch.zeros(expected_size, dtype=torch.bool, device=target_device)
                # Assicurati che gli slice abbiano la stessa dimensione
                n = min(len(values1), len(values2), expected_size)
                base_signals[1:n] = values1[1:n] < values2[1:n]
                
                results = base_signals
            elif self.dna["entry_operator"] == Operator.CROSS_ABOVE.value:
                # Crea segnali con la dimensione corretta
                cross_above = torch.zeros(expected_size, dtype=torch.bool, device=target_device)
                # Assicurati che gli slice abbiano la stessa dimensione
                n = min(len(values1)-1, len(values2)-1, expected_size-1)
                cross_above[1:n+1] = (values1[:n] <= values2[:n]) & (values1[1:n+1] > values2[1:n+1])
                results = cross_above
            elif self.dna["entry_operator"] == Operator.CROSS_BELOW.value:
                # Crea segnali con la dimensione corretta
                cross_below = torch.zeros(expected_size, dtype=torch.bool, device=target_device)
                # Assicurati che gli slice abbiano la stessa dimensione
                n = min(len(values1)-1, len(values2)-1, expected_size-1)
                cross_below[1:n+1] = (values1[:n] >= values2[:n]) & (values1[1:n+1] < values2[1:n+1])
                results = cross_below

            # Applica il filtro di densità
            total_signals = torch.sum(results).item()
            if total_signals == 0:
                return results
                
            # Ottieni e valida i parametri dal config con limiti di sicurezza
            min_bars_between = max(self.config.get("trading.signal_filters.density.min_bars_between", 20), 15)
            max_signals_percent = min(self.config.get("trading.signal_filters.density.max_signals_percent", 0.3), 1)
            # Imposta un limite massimo assoluto al numero di segnali
            max_signals_absolute = 100
            # Calcola max_signals_per_period assicurandosi che sia almeno 1
            signals_from_percent = max(1, int(len(results) * max_signals_percent / 100))
            max_signals_per_period = min(signals_from_percent, max_signals_absolute)
            
            # Log dei parametri di filtro
            logger.debug(f"Signal density parameters:")
            logger.debug(f"- Min bars between signals: {min_bars_between}")
            logger.debug(f"- Max signals percent: {max_signals_percent}%")
            logger.debug(f"- Max signals allowed: {max_signals_per_period}")
            
            # Ottieni gli indici dei segnali e gestisci il caso di tensore 0-d
            signal_indices = torch.nonzero(results)
            if signal_indices.dim() == 0:
                return results
            
            # Converti in lista di indici
            signal_indices = signal_indices.squeeze()
            if signal_indices.dim() == 0:  # Se c'è un solo segnale
                indices_list = [signal_indices.item()]
            else:
                indices_list = signal_indices.tolist()
            
            if not indices_list:
                return results
                
            # Calcola la distanza minima basata sul numero di segnali desiderato
            min_distance = min(min_bars_between, len(results) // max_signals_per_period)
            
            # Seleziona i segnali mantenendo la distanza minima e il numero massimo
            keep_indices = [indices_list[0]]  # Mantieni il primo segnale
            for idx in indices_list[1:]:
                # Verifica sia la distanza minima che il limite massimo
                if idx - keep_indices[-1] >= min_distance and len(keep_indices) < max_signals_per_period:
                    keep_indices.append(idx)
                if len(keep_indices) >= max_signals_per_period:
                    break
            
            # Calcola la correlazione tra i segnali esistenti
            if len(keep_indices) > 1:
                # Usa una finestra di 10 periodi intorno ad ogni segnale
                window_size = 10
                i = len(keep_indices) - 1
                while i > 0:
                    current_idx = keep_indices[i]
                    prev_idx = keep_indices[i-1]
                    
                    # Se i segnali sono troppo vicini, rimuovi il secondo
                    if current_idx - prev_idx < min_bars_between:
                        keep_indices.pop(i)
                        i -= 1
                        continue
                        
                    # Calcola la correlazione tra le finestre
                    if current_idx + window_size < len(values1) and prev_idx + window_size < len(values1):
                        window1 = values1[prev_idx:prev_idx+window_size]
                        window2 = values1[current_idx:current_idx+window_size]
                        correlation = torch.corrcoef(torch.stack([window1, window2]))[0,1]
                        
                        # Se la correlazione è troppo alta, rimuovi il secondo segnale
                        if correlation > 0.8:  # Alta correlazione
                            keep_indices.pop(i)
                    i -= 1
                            
            # Se abbiamo ancora troppi segnali dopo il filtro di correlazione, seleziona in modo casuale
            if len(keep_indices) > max_signals_per_period:
                keep_indices = sorted(random.sample(keep_indices, max_signals_per_period))
            
            # Crea il nuovo tensore di risultati con la dimensione corretta
            new_results = torch.zeros(expected_size, dtype=torch.bool, device=target_device)
            # Assicurati che gli indici siano validi
            valid_indices = [idx for idx in keep_indices if idx < expected_size]
            new_results[valid_indices] = True
            
            # Log del risultato del filtraggio
            logger.info(f"Signal filtering results:")
            logger.info(f"- Original signals: {total_signals}")
            logger.info(f"- Filtered signals: {len(valid_indices)}")
            logger.info(f"- Min distance used: {min_distance}")
            
            results = new_results
                
            # Assicurati che il risultato abbia la dimensione corretta
            if len(results) != len(next(iter(data.values()))):
                logger.error(f"Dimension mismatch: results {len(results)} != data {len(next(iter(data.values())))}")
                return torch.zeros(len(next(iter(data.values()))), dtype=torch.bool)
            
            # Il risultato è già un tensor
            return results
            
        except KeyError as e:
            logger.error(f"KeyError in generate_entry_conditions: {str(e)}")
            logger.error(f"DNA: {self.dna}")
            return torch.zeros(len(next(iter(data.values()))), dtype=torch.bool)
        except Exception as e:
            logger.error(f"Error in generate_entry_conditions: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.zeros(len(next(iter(data.values()))), dtype=torch.bool)

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
                # Usa probabilità dal config per gli operatori
                "entry_operator": np.random.choice(
                    [op.value for op in Operator],
                    p=[
                        self.config.get("trading.signal_filters.operator_weights.greater", 0.15),
                        self.config.get("trading.signal_filters.operator_weights.less", 0.15),
                        self.config.get("trading.signal_filters.operator_weights.cross_above", 0.35),
                        self.config.get("trading.signal_filters.operator_weights.cross_below", 0.35)
                    ]
                ),
                # Usa un rapporto risk/reward più conservativo
                "stop_loss_pct": float(np.random.uniform(
                    self.config.get("trading.mutation.stop_loss.min", 1.5),
                    self.config.get("trading.mutation.stop_loss.max", 2.5)
                )),
                # Take profit è sempre 2-3 volte lo stop loss
                "take_profit_pct": float(np.random.uniform(
                    self.config.get("trading.mutation.stop_loss.min", 1.5),
                    self.config.get("trading.mutation.stop_loss.max", 2.5)
                )) * float(np.random.uniform(2.0, 3.0))
            }
            
            logger.debug(f"Randomized DNA: {self.dna}")
            
        except Exception as e:
            logger.error(f"Error in randomize_dna: {str(e)}")
            self.load_default_dna()

    def load_default_dna(self):
        """Load default DNA configuration from config file"""
        self.dna = {
            "entry_indicator1": "SMA",
            "entry_indicator1_params": {"timeperiod": 21},
            "entry_indicator2": "SMA", 
            "entry_indicator2_params": {"timeperiod": 55},
            "entry_operator": "cross_above",
            "stop_loss_pct": self.config.get("trading.defaults.stop_loss_pct", 2.0),
            "take_profit_pct": self.config.get("trading.defaults.take_profit_pct", 4.0)
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
                            # Mutazione gaussiana con limiti dal config
                            std_dev = self.config.get("trading.mutation.stop_loss.std_dev", 0.5)
                            min_val = self.config.get("trading.mutation.stop_loss.min", 0.5)
                            max_val = self.config.get("trading.mutation.stop_loss.max", 5.0)
                            delta = np.random.normal(0, std_dev)
                            new_value = current + delta
                            self.dna[key] = float(np.clip(new_value, min_val, max_val))
                        elif "take_profit" in key:
                            current = float(self.dna[key])
                            std_dev = self.config.get("trading.mutation.take_profit.std_dev", 1.0)
                            min_val = self.config.get("trading.mutation.take_profit.min", 1.0)
                            max_val = self.config.get("trading.mutation.take_profit.max", 10.0)
                            delta = np.random.normal(0, std_dev)
                            new_value = current + delta
                            self.dna[key] = float(np.clip(new_value, min_val, max_val))
                            
                        # Mantiene take_profit > stop_loss
                        if "stop_loss_pct" in self.dna and "take_profit_pct" in self.dna:
                            if self.dna["take_profit_pct"] <= self.dna["stop_loss_pct"]:
                                multiplier = self.config.get("trading.mutation.take_profit.multiplier", 1.5)
                                self.dna["take_profit_pct"] = self.dna["stop_loss_pct"] * multiplier
                    
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
            child.gene_type = self.gene_type  # Mantiene il tipo di gene del genitore
            
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
