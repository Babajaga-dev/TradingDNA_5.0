# src/models/genes/indicators.py
import numpy as np
import talib

def calculate_bb_upper(x, timeperiod):
    return talib.BBANDS(x, timeperiod=timeperiod)[0]

def calculate_bb_lower(x, timeperiod):
    return talib.BBANDS(x, timeperiod=timeperiod)[2]

def calculate_atr(high, low, close, timeperiod):
    return talib.ATR(high, low, close, timeperiod=timeperiod)

def calculate_stoch(high, low, close, fastk_period, slowk_period, slowd_period):
    slowk, _ = talib.STOCH(high, low, close,
                          fastk_period=fastk_period,
                          slowk_period=slowk_period,
                          slowd_period=slowd_period)
    return slowk

class IndicatorRegistry:
    @staticmethod
    def get_available_indicators():
        return {
            "SMA": {"func": talib.SMA, "params": ["timeperiod"]},
            "EMA": {"func": talib.EMA, "params": ["timeperiod"]},
            "RSI": {"func": talib.RSI, "params": ["timeperiod"]},
            "MACD": {"func": talib.MACD, "params": ["fastperiod", "slowperiod", "signalperiod"]},
            "BB_UPPER": {"func": calculate_bb_upper, "params": ["timeperiod"]},
            "BB_LOWER": {"func": calculate_bb_lower, "params": ["timeperiod"]},
            "CLOSE": {"func": lambda x: x, "params": []}
        }

    @staticmethod
    def generate_random_params(param_names):
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

    @staticmethod
    def calculate_indicator(prices: np.ndarray, indicator: str, params: dict, 
                          available_indicators: dict) -> np.ndarray:
        indicator_info = available_indicators[indicator]
        indicator_func = indicator_info["func"]
        
        valid_params = {}
        for param in indicator_info["params"]:
            if param not in params:
                valid_params[param] = IndicatorRegistry.generate_random_params([param])[param]
            else:
                valid_params[param] = params[param]
        
        if indicator == "CLOSE":
            return prices
        elif indicator == "MACD":
            macd, _, _ = indicator_func(prices, 
                                    fastperiod=valid_params.get("fastperiod", 12),
                                    slowperiod=valid_params.get("slowperiod", 26),
                                    signalperiod=valid_params.get("signalperiod", 9))
            return macd
        else:
            return indicator_func(prices, **valid_params)