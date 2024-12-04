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

class TradingGene:


    def __init__(self, random_init=True):
        """Inizializza un nuovo gene"""
        self.available_indicators = {
            "SMA": {"func": talib.SMA, "params": ["timeperiod"]},
            "EMA": {"func": talib.EMA, "params": ["timeperiod"]},
            "RSI": {"func": talib.RSI, "params": ["timeperiod"]},
            "MACD": {"func": talib.MACD, "params": ["fastperiod", "slowperiod", "signalperiod"]},
            "BB_UPPER": {"func": lambda x, timeperiod: talib.BBANDS(x, timeperiod)[0], "params": ["timeperiod"]},
            "BB_LOWER": {"func": lambda x, timeperiod: talib.BBANDS(x, timeperiod)[2], "params": ["timeperiod"]},
            "CLOSE": {"func": lambda x: x, "params": []}
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
        
        # Filtra i parametri per includere solo quelli accettati dall'indicatore
        valid_params = {k: v for k, v in params.items() if k in indicator_info["params"]}
        
        if indicator == "CLOSE":
            return prices
        elif indicator == "MACD":
            macd, signal, hist = indicator_func(prices, 
                                              fastperiod=valid_params.get("fastperiod", 12),
                                              slowperiod=valid_params.get("slowperiod", 26),
                                              signalperiod=valid_params.get("signalperiod", 9))
            return macd  # Usiamo la linea MACD principale
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
                        'new_value': new_params
                    })
                elif key.endswith('_indicator1') or key.endswith('_indicator2'):
                    old_indicator = self.dna[key]
                    self.dna[key] = np.random.choice(list(self.available_indicators.keys()))
                    # Genera nuovi parametri appropriati per il nuovo indicatore
                    params_key = f"{key}_params"
                    self.dna[params_key] = self._generate_random_params(
                        self.available_indicators[self.dna[key]]["params"]
                    )
                    mutations.append({
                        'type': 'indicator',
                        'key': key,
                        'old_value': old_indicator,
                        'new_value': self.dna[key]
                    })
                elif key.endswith('_operator'):
                    old_operator = self.dna[key]
                    self.dna[key] = np.random.choice(list(Operator))
                    mutations.append({
                        'type': 'operator',
                        'key': key,
                        'old_value': old_operator,
                        'new_value': self.dna[key]
                    })
                
                elif key == "position_size_pct":
                    old_size = self.dna[key]
                    new_size = self.dna[key] * np.random.uniform(0.8, 1.2)
                    self.dna[key] = max(1.0, min(20.0, new_size))
                    mutations.append({
                        'type': 'position_size',
                        'key': key,
                        'old_value': old_size,
                        'new_value': self.dna[key],
                        'mutation_factor': self.dna[key] / old_size
                    })
                
                elif key == "stop_loss_pct":
                    old_sl = self.dna[key]
                    new_sl = self.dna[key] * np.random.uniform(0.8, 1.2)
                    self.dna[key] = max(0.5, min(5.0, new_sl))
                    mutations.append({
                        'type': 'stop_loss',
                        'key': key,
                        'old_value': old_sl,
                        'new_value': self.dna[key],
                        'mutation_factor': self.dna[key] / old_sl
                    })
                
                elif key == "take_profit_pct":
                    old_tp = self.dna[key]
                    new_tp = self.dna[key] * np.random.uniform(0.8, 1.2)
                    self.dna[key] = max(1.0, min(10.0, new_tp))
                    mutations.append({
                        'type': 'take_profit',
                        'key': key,
                        'old_value': old_tp,
                        'new_value': self.dna[key],
                        'mutation_factor': self.dna[key] / old_tp
                    })

        # Se ci sono state mutazioni, logga i dettagli
        if mutations:
            print("\n🧬 MUTAZIONE GENE")
            print(f"Tasso di mutazione: {mutation_rate*100:.1f}%")
            print("\nModifiche applicate:")
            
            for mut in mutations:
                if mut['type'] in ['position_size', 'stop_loss', 'take_profit']:
                    print(f"\n  {mut['key']}:")
                    print(f"    Vecchio valore: {mut['old_value']:.2f}%")
                    print(f"    Nuovo valore: {mut['new_value']:.2f}%")
                    print(f"    Fattore mutazione: {mut['mutation_factor']:.2f}x")
                
                elif mut['type'] == 'parameter':
                    print(f"\n  {mut['key']}:")
                    print(f"    Vecchio valore: {mut['old_value']}")
                    print(f"    Nuovo valore: {mut['new_value']}")
                    if mut['mutation_factor'] != 'N/A':
                        print(f"    Fattore mutazione: {mut['mutation_factor']:.2f}x")
                
                elif mut['type'] in ['indicator', 'operator']:
                    print(f"\n  {mut['key']}:")
                    print(f"    Vecchio valore: {mut['old_value']}")
                    print(f"    Nuovo valore: {mut['new_value']}")
            
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

    def crossover(self, other: 'TradingGene') -> 'TradingGene':
        """Esegue il crossover con un altro gene"""
        child = TradingGene()
        for key in self.dna:
            if np.random.random() < 0.5:
                child.dna[key] = self.dna[key]
            else:
                child.dna[key] = other.dna[key]
        return child
