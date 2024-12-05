# src/models/genes/torch_gene.py
import torch
from typing import Dict
import numpy as np

from .base import TradingGene
from .indicators import IndicatorRegistry

class TorchIndicators:
    @staticmethod
    def bollinger_bands(data: torch.Tensor, window: int, num_std: float = 2.0):
        ma = TorchIndicators.moving_average(data, window)
        rolling_std = torch.stack([data[i:i+window].std() 
                                 for i in range(len(data)-window+1)])
        rolling_std = torch.cat([torch.zeros(window-1), rolling_std])
        upper_band = ma + (rolling_std * num_std)
        lower_band = ma - (rolling_std * num_std)
        return upper_band, ma, lower_band

    @staticmethod
    def moving_average(data: torch.Tensor, window: int):
        weights = torch.ones(window).float() / window
        return torch.conv1d(
            data.view(1, 1, -1), 
            weights.view(1, 1, -1), 
            padding=window-1
        ).view(-1)[-len(data):]

    @staticmethod
    def exponential_moving_average(data: torch.Tensor, span: int):
        alpha = 2.0 / (span + 1)
        weights = (1 - alpha) ** torch.arange(span - 1, -1, -1).float()
        weights = weights / weights.sum()
        return torch.conv1d(
            data.view(1, 1, -1),
            weights.view(1, 1, -1),
            padding=span-1
        ).view(-1)[-len(data):]

    @staticmethod
    def rsi(data: torch.Tensor, window: int):
        delta = data[1:] - data[:-1]
        gains = torch.where(delta > 0, delta, torch.zeros_like(delta))
        losses = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        avg_gains = TorchIndicators.moving_average(gains, window)
        avg_losses = TorchIndicators.moving_average(losses, window)
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return torch.cat([torch.zeros(1), rsi])

    @staticmethod
    def macd(data: torch.Tensor, fast_period: int = 12, 
             slow_period: int = 26, signal_period: int = 9):
        fast_ema = TorchIndicators.exponential_moving_average(data, fast_period)
        slow_ema = TorchIndicators.exponential_moving_average(data, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = TorchIndicators.exponential_moving_average(macd_line, signal_period)
        return macd_line

class TorchGene(TradingGene):
    def __init__(self, random_init=True):
        self.available_indicators = {
            "SMA": {"func": TorchIndicators.moving_average, "params": ["window"]},
            "EMA": {"func": TorchIndicators.exponential_moving_average, "params": ["span"]},
            "RSI": {"func": TorchIndicators.rsi, "params": ["window"]},
            "MACD": {"func": TorchIndicators.macd, 
                    "params": ["fast_period", "slow_period", "signal_period"]},
            "BB_UPPER": {"func": lambda x, window: TorchIndicators.bollinger_bands(x, window)[0], 
                        "params": ["window"]},
            "BB_LOWER": {"func": lambda x, window: TorchIndicators.bollinger_bands(x, window)[2], 
                        "params": ["window"]},
            "CLOSE": {"func": lambda x: x, "params": []}
        }
        super().__init__(random_init)

    def calculate_indicator(self, prices: np.ndarray, indicator: str, params: Dict) -> np.ndarray:
        prices_tensor = torch.tensor(prices, dtype=torch.float32)
        
        valid_params = {}
        for param_name, value in params.items():
            if param_name == "timeperiod":
                valid_params["window"] = value
            elif param_name in self.available_indicators[indicator]["params"]:
                valid_params[param_name] = value
        
        result = self.available_indicators[indicator]["func"](prices_tensor, **valid_params)
        return result.numpy()