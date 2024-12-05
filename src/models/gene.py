import torch
from typing import Dict, Optional, Union
from datetime import datetime
from enum import Enum
import numpy as np
import talib
from src.models.common import Signal, SignalType, MarketData
from src.utils.config import config

class Operator(Enum):
    GREATER = ">"
    LESS = "<"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"

def calculate_bb_upper(x, timeperiod):
    return talib.BBANDS(x, timeperiod=timeperiod)[0]

def calculate_bb_lower(x, timeperiod):
    return talib.BBANDS(x, timeperiod=timeperiod)[2]

def calculate_close(x):
    return x

from datetime import datetime
from typing import List, Dict
from enum import Enum
import numpy as np
import talib
from src.models.common import Signal, SignalType, MarketData
from src.utils.config import config

class Operator(Enum):
    GREATER = ">"
    LESS = "<"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"

def calculate_bb_upper(x, timeperiod):
    return talib.BBANDS(x, timeperiod=timeperiod)[0]

def calculate_bb_lower(x, timeperiod):
    return talib.BBANDS(x, timeperiod=timeperiod)[2]

def calculate_close(x):
    return x

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int):
    """Calcola l'Average True Range"""
    return talib.ATR(high, low, close, timeperiod=timeperiod)

def calculate_stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   fastk_period: int, slowk_period: int, slowd_period: int):
    """Calcola l'oscillatore stocastico"""
    slowk, slowd = talib.STOCH(high, low, close, 
                              fastk_period=fastk_period,
                              slowk_period=slowk_period,
                              slowd_period=slowd_period)
    return slowk

class TradingGene:

    def __init__(self, random_init=True):
        """Inizializza un nuovo gene"""
        self.available_indicators = {
            "SMA": {"func": talib.SMA, "params": ["timeperiod"]},
            "EMA": {"func": talib.EMA, "params": ["timeperiod"]},
            "RSI": {"func": talib.RSI, "params": ["timeperiod"]},
            "MACD": {"func": talib.MACD, "params": ["fastperiod", "slowperiod", "signalperiod"]},
            "BB_UPPER": {"func": calculate_bb_upper, "params": ["timeperiod"]},
            "BB_LOWER": {"func": calculate_bb_lower, "params": ["timeperiod"]},
            "CLOSE": {"func": calculate_close, "params": []}
        }

        self.fitness_score = None
        self.performance_history = {}
        self.mutation_history = []
        
        if random_init:
            # Prima selezioniamo gli indicatori
            entry_ind1 = np.random.choice(list(self.available_indicators.keys()))
            entry_ind2 = np.random.choice(list(self.available_indicators.keys()))
            exit_ind1 = np.random.choice(list(self.available_indicators.keys()))
            exit_ind2 = np.random.choice(list(self.available_indicators.keys()))
            
            # Poi creiamo il DNA completo
            self.dna = {
                # Indicatori di entrata
                "entry_indicator1": entry_ind1,
                "entry_indicator1_params": self._generate_random_params(self.available_indicators[entry_ind1]["params"]),
                "entry_indicator2": entry_ind2,
                "entry_indicator2_params": self._generate_random_params(self.available_indicators[entry_ind2]["params"]),
                "entry_operator": np.random.choice(list(Operator)),
                
                # Indicatori di uscita
                "exit_indicator1": exit_ind1,
                "exit_indicator1_params": self._generate_random_params(self.available_indicators[exit_ind1]["params"]),
                "exit_indicator2": exit_ind2,
                "exit_indicator2_params": self._generate_random_params(self.available_indicators[exit_ind2]["params"]),
                "exit_operator": np.random.choice(list(Operator)),
                
                # Parametri di risk management
                "stop_loss_pct": np.random.uniform(0.5, 5.0),
                "take_profit_pct": np.random.uniform(1.0, 10.0),
                "position_size_pct": np.random.uniform(1.0, 20.0)
            }
        else:
            # Carica i parametri dal file di configurazione
            self.dna = {
                # Indicatori di entrata
                "entry_indicator1": config.get("trading.indicators.entry.indicator1.type", "SMA"),
                "entry_indicator1_params": config.get("trading.indicators.entry.indicator1.params", {"timeperiod": 20}),
                "entry_indicator2": config.get("trading.indicators.entry.indicator2.type", "SMA"),
                "entry_indicator2_params": config.get("trading.indicators.entry.indicator2.params", {"timeperiod": 50}),
                "entry_operator": Operator(config.get("trading.indicators.entry.operator", "cross_above")),
                
                # Indicatori di uscita
                "exit_indicator1": config.get("trading.indicators.exit.indicator1.type", "RSI"),
                "exit_indicator1_params": config.get("trading.indicators.exit.indicator1.params", {"timeperiod": 14}),
                "exit_indicator2": config.get("trading.indicators.exit.indicator2.type", "CLOSE"),
                "exit_indicator2_params": config.get("trading.indicators.exit.indicator2.params", {}),
                "exit_operator": Operator(config.get("trading.indicators.exit.operator", ">")),
                
                # Parametri di risk management
                "stop_loss_pct": config.get("trading.position.stop_loss_pct", 2.0),
                "take_profit_pct": config.get("trading.position.take_profit_pct", 4.0),
                "position_size_pct": config.get("trading.position.size_pct", 5.0)
            }

    def initialize_random(self):
        """Inizializza il gene con valori casuali"""
        self.dna = {
            # Indicatori di entrata
            "entry_indicator1": np.random.choice(list(self.available_indicators.keys())),
            "entry_indicator1_params": {
                "timeperiod": np.random.randint(5, 200)
            },
            "entry_indicator2": np.random.choice(list(self.available_indicators.keys())),
            "entry_indicator2_params": {
                "timeperiod": np.random.randint(5, 200)
            },
            "entry_operator": np.random.choice(list(Operator)),
            
            # Indicatori di uscita
            "exit_indicator1": np.random.choice(list(self.available_indicators.keys())),
            "exit_indicator1_params": {
                "timeperiod": np.random.randint(5, 200)
            },
            "exit_indicator2": np.random.choice(list(self.available_indicators.keys())),
            "exit_indicator2_params": {
                "timeperiod": np.random.randint(5, 200)
            },
            "exit_operator": np.random.choice(list(Operator)),
            
            # Parametri di risk management
            "stop_loss_pct": np.random.uniform(0.5, 5.0),
            "take_profit_pct": np.random.uniform(1.0, 10.0),
            "position_size_pct": np.random.uniform(1.0, 20.0)
        }
 
    def _generate_random_params(self, param_names):
        """Genera parametri casuali per un indicatore"""
        params = {}
        for param in param_names:
            if param == "timeperiod":
                params[param] = np.random.randint(5, 200)
            elif param == "fastperiod":
                params[param] = np.random.randint(5, 50)
            elif param == "slowperiod":
                params[param] = np.random.randint(10, 100)
            elif param == "signalperiod":
                params[param] = np.random.randint(5, 30)
        return params

    def calculate_indicator(self, prices: np.ndarray, indicator: str, params: Dict) -> np.ndarray:
        """Calcola un indicatore sui prezzi"""
        indicator_info = self.available_indicators[indicator]
        indicator_func = indicator_info["func"]
        
        # Assicurati che tutti i parametri richiesti siano presenti
        required_params = indicator_info["params"]
        valid_params = {}
        
        for param in required_params:
            if param not in params:
                # Se manca un parametro richiesto, genera un valore casuale
                valid_params[param] = self._generate_random_params([param])[param]
            else:
                valid_params[param] = params[param]
        
        if indicator == "CLOSE":
            return prices
        elif indicator == "MACD":
            macd, signal, hist = indicator_func(prices, 
                                              fastperiod=valid_params.get("fastperiod", 12),
                                              slowperiod=valid_params.get("slowperiod", 26),
                                              signalperiod=valid_params.get("signalperiod", 9))
            return macd
        elif indicator in ["BB_UPPER", "BB_LOWER"]:
            # Assicurati che timeperiod sia presente per le Bollinger Bands
            if "timeperiod" not in valid_params:
                valid_params["timeperiod"] = 20  # valore di default
            return indicator_func(prices, timeperiod=valid_params["timeperiod"])
        else:
            return indicator_func(prices, **valid_params)

    def mutate(self, mutation_rate: float = 0.1):
        """Muta il DNA del gene"""
        mutations = []
        for key in self.dna:
            if np.random.random() < mutation_rate:
                if key.endswith('_params'):
                    indicator_key = key.replace('_params', '')
                    indicator = self.dna[indicator_key]
                    new_params = self._generate_random_params(
                        self.available_indicators[indicator]["params"]
                    )
                    old_params = self.dna[key].copy()
                    self.dna[key] = new_params
                    mutations.append({
                        'type': 'parameter',
                        'key': key,
                        'old_value': old_params,
                        'new_value': new_params,
                        'mutation_factor': 'N/A'  # Add default mutation_factor
                    })
                elif key.endswith('_indicator1') or key.endswith('_indicator2'):
                    old_indicator = self.dna[key]
                    self.dna[key] = np.random.choice(list(self.available_indicators.keys()))
                    params_key = f"{key}_params"
                    self.dna[params_key] = self._generate_random_params(
                        self.available_indicators[self.dna[key]]["params"]
                    )
                    mutations.append({
                        'type': 'indicator',
                        'key': key,
                        'old_value': old_indicator,
                        'new_value': self.dna[key],
                        'mutation_factor': 'N/A'  # Add default mutation_factor
                    })
                elif key.endswith('_operator'):
                    old_operator = self.dna[key]
                    self.dna[key] = np.random.choice(list(Operator))
                    mutations.append({
                        'type': 'operator',
                        'key': key,
                        'old_value': old_operator,
                        'new_value': self.dna[key],
                        'mutation_factor': 'N/A'  # Add default mutation_factor
                    })
                elif key in ["position_size_pct", "stop_loss_pct", "take_profit_pct"]:
                    old_value = self.dna[key]
                    mutation_factor = np.random.uniform(0.8, 1.2)
                    new_value = old_value * mutation_factor
                    
                    # Apply limits based on parameter type
                    if key == "position_size_pct":
                        new_value = max(1.0, min(20.0, new_value))
                    elif key == "stop_loss_pct":
                        new_value = max(0.5, min(5.0, new_value))
                    elif key == "take_profit_pct":
                        new_value = max(1.0, min(10.0, new_value))
                    
                    self.dna[key] = new_value
                    mutations.append({
                        'type': key.replace('_pct', ''),
                        'key': key,
                        'old_value': old_value,
                        'new_value': new_value,
                        'mutation_factor': mutation_factor
                    })

        # Se ci sono state mutazioni, logga i dettagli
        if mutations:
            print("\n🧬 MUTAZIONE GENE")
            print(f"Tasso di mutazione: {mutation_rate*100:.1f}%")
            print("\nModifiche applicate:")
            
            for mut in mutations:
                mut_type = mut['type']
                if mut_type in ['position_size', 'stop_loss', 'take_profit']:
                    print(f"\n  {mut['key']}:")
                    print(f"    Vecchio valore: {mut['old_value']:.2f}%")
                    print(f"    Nuovo valore: {mut['new_value']:.2f}%")
                    print(f"    Fattore mutazione: {mut['mutation_factor']:.2f}x")
                else:
                    print(f"\n  {mut['key']}:")
                    print(f"    Vecchio valore: {mut['old_value']}")
                    print(f"    Nuovo valore: {mut['new_value']}")
                    if mut['mutation_factor'] != 'N/A':
                        print(f"    Fattore mutazione: {mut['mutation_factor']:.2f}x")
            
            # Aggiungi questa mutazione alla storia
            self.mutation_history.append({
                'timestamp': datetime.now(),
                'mutations': mutations
            })
            
        # Verifica dopo la mutazione
        self.validate_dna()

    def get_mutation_history(self):
        """Restituisce la storia delle mutazioni in formato leggibile"""
        if not self.mutation_history:
            return "Nessuna mutazione registrata"
            
        history = []
        for record in self.mutation_history:
            history.append({
                'timestamp': record['timestamp'],
                'changes': [f"{m['key']}: {m['old_value']} -> {m['new_value']}" 
                          for m in record['mutations']]
            })
        return history

    def check_condition(self, data1: np.ndarray, operator: Operator, data2: np.ndarray) -> bool:
        """Verifica una condizione tra due serie di dati"""
        if operator == Operator.GREATER:
            return data1[-1] > data2[-1]
        elif operator == Operator.LESS:
            return data1[-1] < data2[-1]
        elif operator == Operator.CROSS_ABOVE:
            return data1[-2] <= data2[-2] and data1[-1] > data2[-1]
        elif operator == Operator.CROSS_BELOW:
            return data1[-2] >= data2[-2] and data1[-1] < data2[-1]
        return False

    def generate_signals(self, market_data: List[MarketData]) -> List[Signal]:
        """Genera segnali basati sulle condizioni del gene"""
        signals = []
        min_candles = config.get("simulator.min_candles", 50)
        
        if len(market_data) < min_candles:  # Minimo di dati necessari
            return signals

        # Converti i dati in array numpy per gli indicatori
        prices = np.array([d.close for d in market_data])
        
        # Calcola gli indicatori di entrata
        entry_ind1 = self.calculate_indicator(
            prices, 
            self.dna["entry_indicator1"], 
            self.dna["entry_indicator1_params"]
        )
        entry_ind2 = self.calculate_indicator(
            prices,
            self.dna["entry_indicator2"],
            self.dna["entry_indicator2_params"]
        )
        
        # Calcola gli indicatori di uscita
        exit_ind1 = self.calculate_indicator(
            prices,
            self.dna["exit_indicator1"],
            self.dna["exit_indicator1_params"]
        )
        exit_ind2 = self.calculate_indicator(
            prices,
            self.dna["exit_indicator2"],
            self.dna["exit_indicator2_params"]
        )
        
        # Gestisce i NaN
        if np.isnan(entry_ind1[-1]) or np.isnan(entry_ind2[-1]) or \
           np.isnan(exit_ind1[-1]) or np.isnan(exit_ind2[-1]):
            return signals

        # Verifica condizioni di entrata
        if self.check_condition(entry_ind1, self.dna["entry_operator"], entry_ind2):
            current_price = market_data[-1].close
            stop_loss = current_price * (1 - self.dna["stop_loss_pct"]/100)
            take_profit = current_price * (1 + self.dna["take_profit_pct"]/100)
            
            signals.append(Signal(
                type=SignalType.LONG,
                timestamp=market_data[-1].timestamp,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={"gene_id": id(self)}
            ))
        
        # Verifica condizioni di uscita
        elif self.check_condition(exit_ind1, self.dna["exit_operator"], exit_ind2):
            signals.append(Signal(
                type=SignalType.EXIT,
                timestamp=market_data[-1].timestamp,
                price=market_data[-1].close,
                metadata={"gene_id": id(self)}
            ))
            
        return signals

    def validate_dna(self):
        """Valida il DNA del gene e sistema eventuali valori non validi"""
        # Valida gli indicatori
        for key in ['entry_indicator1', 'entry_indicator2', 'exit_indicator1', 'exit_indicator2']:
            if self.dna[key] not in self.available_indicators:
                self.dna[key] = np.random.choice(list(self.available_indicators.keys()))
                self.dna[f'{key}_params'] = self._generate_random_params(
                    self.available_indicators[self.dna[key]]["params"]
                )
        
        # Valida i parametri degli indicatori
        for key in ['entry_indicator1_params', 'entry_indicator2_params', 
                   'exit_indicator1_params', 'exit_indicator2_params']:
            base_key = key.replace('_params', '')
            indicator = self.dna[base_key]
            required_params = self.available_indicators[indicator]["params"]
            
            # Se mancano parametri richiesti, genera nuovi parametri
            if not all(param in self.dna[key] for param in required_params):
                self.dna[key] = self._generate_random_params(required_params)
            
            # Rimuovi parametri non necessari
            self.dna[key] = {k: v for k, v in self.dna[key].items() if k in required_params}
        
        # Valida gli operatori
        if not isinstance(self.dna['entry_operator'], Operator):
            self.dna['entry_operator'] = np.random.choice(list(Operator))
        if not isinstance(self.dna['exit_operator'], Operator):
            self.dna['exit_operator'] = np.random.choice(list(Operator))
        
        # Valida e correggi i parametri di risk management
        self.dna['stop_loss_pct'] = np.clip(self.dna['stop_loss_pct'], 0.5, 5.0)
        self.dna['take_profit_pct'] = np.clip(self.dna['take_profit_pct'], 1.0, 10.0)
        self.dna['position_size_pct'] = np.clip(self.dna['position_size_pct'], 1.0, 20.0)
        
        # Assicura che il take profit sia maggiore dello stop loss
        if self.dna['take_profit_pct'] <= self.dna['stop_loss_pct']:
            self.dna['take_profit_pct'] = self.dna['stop_loss_pct'] * 2

    def crossover(self, other: 'TradingGene') -> 'TradingGene':
        """Esegue il crossover con un altro gene"""
        child = TradingGene()
        for key in self.dna:
            if np.random.random() < 0.5:
                child.dna[key] = self.dna[key]
            else:
                child.dna[key] = other.dna[key]
        return child

class TorchIndicators:
    
    @staticmethod
    def bollinger_bands(data: torch.Tensor, window: int, num_std: float = 2.0) -> torch.Tensor:
        """Calcola le Bollinger Bands usando PyTorch"""
        # Calcola media mobile
        ma = TorchIndicators.moving_average(data, window)
        
        # Calcola deviazione standard
        rolling_std = torch.stack([data[i:i+window].std() 
                                 for i in range(len(data)-window+1)])
        rolling_std = torch.cat([torch.zeros(window-1), rolling_std])
        
        # Calcola le bande
        upper_band = ma + (rolling_std * num_std)
        lower_band = ma - (rolling_std * num_std)
        
        return upper_band, ma, lower_band

    @staticmethod
    def get_bb_upper(data: torch.Tensor, window: int) -> torch.Tensor:
        """Restituisce la banda superiore delle Bollinger Bands"""
        upper, _, _ = TorchIndicators.bollinger_bands(data, window)
        return upper

    @staticmethod
    def get_bb_lower(data: torch.Tensor, window: int) -> torch.Tensor:
        """Restituisce la banda inferiore delle Bollinger Bands"""
        _, _, lower = TorchIndicators.bollinger_bands(data, window)
        return lower

    @staticmethod
    def moving_average(data: torch.Tensor, window: int) -> torch.Tensor:
        """Calcola la media mobile usando PyTorch"""
        weights = torch.ones(window).float() / window
        return torch.conv1d(
            data.view(1, 1, -1), 
            weights.view(1, 1, -1), 
            padding=window-1
        ).view(-1)[-len(data):]

    @staticmethod
    def exponential_moving_average(data: torch.Tensor, span: int) -> torch.Tensor:
        """Calcola l'EMA usando PyTorch"""
        alpha = 2.0 / (span + 1)
        weights = (1 - alpha) ** torch.arange(span - 1, -1, -1).float()
        weights = weights / weights.sum()
        
        # Usa conv1d per il calcolo efficiente
        return torch.conv1d(
            data.view(1, 1, -1),
            weights.view(1, 1, -1),
            padding=span-1
        ).view(-1)[-len(data):]

    @staticmethod
    def rsi(data: torch.Tensor, window: int) -> torch.Tensor:
        """Calcola l'RSI usando PyTorch"""
        # Calcola le differenze
        delta = data[1:] - data[:-1]
        
        # Separa guadagni e perdite
        gains = torch.where(delta > 0, delta, torch.zeros_like(delta))
        losses = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        # Calcola medie mobili di guadagni e perdite
        avg_gains = TorchIndicators.moving_average(gains, window)
        avg_losses = TorchIndicators.moving_average(losses, window)
        
        # Calcola RS e RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Evita divisione per zero
        rsi = 100 - (100 / (1 + rs))
        
        # Padding per mantenere la lunghezza originale
        return torch.cat([torch.zeros(1), rsi])

    @staticmethod
    def macd(data: torch.Tensor, fast_period: int = 12, 
             slow_period: int = 26, signal_period: int = 9) -> torch.Tensor:
        """Calcola il MACD usando PyTorch"""
        # Calcola EMA veloce e lenta
        fast_ema = TorchIndicators.exponential_moving_average(data, fast_period)
        slow_ema = TorchIndicators.exponential_moving_average(data, slow_period)
        
        # Calcola linea MACD
        macd_line = fast_ema - slow_ema
        
        # Calcola linea del segnale
        signal_line = TorchIndicators.exponential_moving_average(macd_line, signal_period)
        
        return macd_line




class TorchGene(TradingGene):

    def __init__(self, random_init=True):
        super().__init__(random_init)
        self.available_indicators = {
            "SMA": {"func": TorchIndicators.moving_average, "params": ["window"]},
            "EMA": {"func": TorchIndicators.exponential_moving_average, "params": ["span"]},
            "RSI": {"func": TorchIndicators.rsi, "params": ["window"]},
            "MACD": {"func": TorchIndicators.macd, 
                    "params": ["fast_period", "slow_period", "signal_period"]},
            "BB_UPPER": {"func": TorchIndicators.get_bb_upper, 
                        "params": ["window"]},
            "BB_LOWER": {"func": TorchIndicators.get_bb_lower, 
                        "params": ["window"]},
            "CLOSE": {"func": lambda x: x, "params": []}
        }

    def calculate_indicator(self, prices: np.ndarray, indicator: str, params: Dict) -> np.ndarray:
        """Calcola un indicatore sui prezzi usando PyTorch"""
        # Converti numpy array in tensor
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        
        indicator_info = self.available_indicators[indicator]
        indicator_func = indicator_info["func"]
        
        # Filtra i parametri validi e rinomina timeperiod in window se necessario
        valid_params = {}
        for param_name, value in params.items():
            if param_name == "timeperiod":
                valid_params["window"] = value
            elif param_name in indicator_info["params"]:
                valid_params[param_name] = value
        
        # Calcola l'indicatore
        if indicator == "MACD":
            result = indicator_func(prices_tensor,
                                 fast_period=valid_params.get("fast_period", 12),
                                 slow_period=valid_params.get("slow_period", 26),
                                 signal_period=valid_params.get("signal_period", 9))
        elif indicator == "CLOSE":
            result = prices_tensor
        else:
            result = indicator_func(prices_tensor, **valid_params)
            
        # Converti il risultato in numpy array
        return result.numpy()


class VolatilityAdaptiveGene(TradingGene):
    """Gene che adatta la size delle posizioni in base alla volatilità"""
    
    def __init__(self, random_init=True):
        super().__init__(random_init=False)
        
        self.available_indicators.update({
            "ATR": {"func": calculate_atr, "params": ["timeperiod"]}
        })
        
        # Carica parametri dalla configurazione
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
        
        # Usa limiti dalla configurazione
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

class MomentumGene(TradingGene):
    """Gene che utilizza indicatori di momentum per le decisioni di trading"""
    
    def __init__(self, random_init=True):
        super().__init__(random_init=False)
        
        self.available_indicators.update({
            "STOCH": {"func": calculate_stoch, "params": ["fastk_period", "slowk_period", "slowd_period"]},
            "RSI": {"func": talib.RSI, "params": ["timeperiod"]},
            "ADX": {"func": talib.ADX, "params": ["timeperiod"]}
        })
        
        params = config.get("trading.momentum_gene.parameters", {})
        
        if random_init:
            self.initialize_momentum_dna(params)
        else:
            self.dna.update({
                "momentum_threshold": params.get("momentum_threshold", {}).get("default", 70),
                "trend_strength_threshold": params.get("trend_strength", {}).get("default", 25),
                "overbought_level": params.get("overbought_level", {}).get("default", 80),
                "oversold_level": params.get("oversold_level", {}).get("default", 20)
            })

    def initialize_momentum_dna(self, params):
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
        base_signals = super().generate_signals(market_data)
        
        if base_signals and base_signals[0].type in [SignalType.LONG, SignalType.SHORT]:
            if not self.check_momentum_conditions(market_data):
                return []
                
        return base_signals

class PatternRecognitionGene(TradingGene):
    """Gene che utilizza il riconoscimento di pattern candlestick"""
    
    def __init__(self, random_init=True):
        super().__init__(random_init=False)
        
        # Mappa per la conversione dei nomi dei pattern
        pattern_name_map = {
            "ENGULFING": "CDLENGULFING",
            "HAMMER": "CDLHAMMER",
            "DOJI": "CDLDOJI",
            "EVENINGSTAR": "CDLEVENINGSTAR",
            "MORNINGSTAR": "CDLMORNINGSTAR",
            "HARAMI": "CDLHARAMI",
            "SHOOTINGSTAR": "CDLSHOOTINGSTAR",
            "MARUBOZU": "CDLMARUBOZU"
        }
        
        # Carica pattern dalla configurazione con correzione dei nomi
        enabled_patterns = config.get("trading.pattern_gene.patterns", [])
        self.available_patterns = {}
        
        for pattern in enabled_patterns:
            if pattern in pattern_name_map:
                talib_name = pattern_name_map[pattern]
                if hasattr(talib, talib_name):
                    self.available_patterns[pattern] = (
                        getattr(talib, talib_name),
                        2 if pattern in ["ENGULFING", "EVENINGSTAR", "MORNINGSTAR"] else 1
                    )
                else:
                    print(f"Warning: Pattern {pattern} ({talib_name}) not found in TA-Lib")
        
        params = config.get("trading.pattern_gene.parameters", {})
        
        if random_init:
            self.initialize_pattern_dna(params)
        else:
            self.dna.update({
                "required_patterns": params.get("required_patterns", {}).get("default", 2),
                "pattern_window": params.get("pattern_window", {}).get("default", 3),
                "confirmation_periods": params.get("confirmation_periods", {}).get("default", 1)
            })

    def initialize_pattern_dna(self, params):
        self.dna.update({
            "required_patterns": np.random.randint(
                params.get("required_patterns", {}).get("min", 1),
                params.get("required_patterns", {}).get("max", 3)
            ),
            "pattern_window": np.random.randint(
                params.get("pattern_window", {}).get("min", 2),
                params.get("pattern_window", {}).get("max", 5)
            ),
            "confirmation_periods": np.random.randint(
                params.get("confirmation_periods", {}).get("min", 1),
                params.get("confirmation_periods", {}).get("max", 3)
            )
        })

    def detect_patterns(self, market_data: List[MarketData]) -> Dict[str, int]:
        """Rileva pattern candlestick nei dati di mercato"""
        if len(market_data) < 10:  # Minimo di candele necessarie
            return {}
            
        opens = np.array([d.open for d in market_data])
        highs = np.array([d.high for d in market_data])
        lows = np.array([d.low for d in market_data])
        closes = np.array([d.close for d in market_data])
        
        patterns = {}
        for name, (func, _) in self.available_patterns.items():
            result = func(opens, highs, lows, closes)
            if not np.isnan(result[-1]):
                patterns[name] = result[-1]
                
        return patterns

    def generate_signals(self, market_data: List[MarketData]) -> List[Signal]:
        """Genera segnali basati sui pattern rilevati"""
        base_signals = super().generate_signals(market_data)
        
        if base_signals and base_signals[0].type in [SignalType.LONG, SignalType.SHORT]:
            patterns = self.detect_patterns(market_data)
            bullish_patterns = sum(1 for v in patterns.values() if v > 0)
            bearish_patterns = sum(1 for v in patterns.values() if v < 0)
            
            # Verifica la conferma dei pattern
            signal_type = base_signals[0].type
            if (signal_type == SignalType.LONG and 
                bullish_patterns < self.dna["required_patterns"]):
                return []
            elif (signal_type == SignalType.SHORT and 
                  bearish_patterns < self.dna["required_patterns"]):
                return []
                
        return base_signals

def create_ensemble_gene(random_init=True):
    """Crea un ensemble di geni specializzati"""
    volatility_gene = VolatilityAdaptiveGene(random_init)
    momentum_gene = MomentumGene(random_init)
    pattern_gene = PatternRecognitionGene(random_init)
    base_gene = TradingGene(random_init)
    
    return [base_gene, volatility_gene, momentum_gene, pattern_gene]

