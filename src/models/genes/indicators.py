# src/models/genes/indicators.py
import numpy as np
import talib
import logging
from typing import List
from ..common import MarketData
from ...utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
config = Config()

def calculate_bb_upper(x: np.ndarray, timeperiod: int) -> np.ndarray:
    try:
        upper, _, _ = talib.BBANDS(x, timeperiod=timeperiod)
        return upper
    except Exception as e:
        logger.error(f"Errore BB_UPPER: {str(e)}")
        return np.full_like(x, np.nan)

def calculate_bb_lower(x: np.ndarray, timeperiod: int) -> np.ndarray:
    try:
        _, _, lower = talib.BBANDS(x, timeperiod=timeperiod)
        return lower
    except Exception as e:
        logger.error(f"Errore BB_LOWER: {str(e)}")
        return np.full_like(x, np.nan)

def calculate_sma(x: np.ndarray, timeperiod: int) -> np.ndarray:
    try:
        if len(x) < timeperiod:
            return np.full_like(x, np.nan)
        return talib.SMA(x, timeperiod=timeperiod)
    except Exception as e:
        logger.error(f"Errore SMA: {str(e)}")
        return np.full_like(x, np.nan)

def calculate_ema(x: np.ndarray, timeperiod: int) -> np.ndarray:
    try:
        if len(x) < timeperiod:
            return np.full_like(x, np.nan)
        alpha_multiplier = config.get("trading.indicators.parameters.ema.alpha_multiplier", 2.0)
        alpha = alpha_multiplier / (timeperiod + 1)
        return talib.EMA(x, timeperiod=timeperiod) * alpha
    except Exception as e:
        logger.error(f"Errore EMA: {str(e)}")
        return np.full_like(x, np.nan)

def calculate_rsi(x: np.ndarray, timeperiod: int) -> np.ndarray:
    try:
        if len(x) < timeperiod + 1:
            return np.full_like(x, np.nan)
        
        # Ottieni parametri RSI dal config
        rsi_params = config.get("trading.indicators.parameters.rsi", {})
        epsilon = rsi_params.get("epsilon", 1e-10)
        scale = rsi_params.get("scale", 100.0)
        
        # Calcola RSI con parametri configurati
        delta = np.diff(x)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gains = np.convolve(gains, np.ones(timeperiod)/timeperiod, mode='valid')
        avg_losses = np.convolve(losses, np.ones(timeperiod)/timeperiod, mode='valid')
        
        rs = avg_gains / (avg_losses + epsilon)
        rsi = scale - (scale / (1 + rs))
        
        # Padding per mantenere la lunghezza originale
        padding = np.full(timeperiod, np.nan)
        return np.concatenate([padding, rsi])
    except Exception as e:
        logger.error(f"Errore RSI: {str(e)}")
        return np.full_like(x, np.nan)

def calculate_macd(x: np.ndarray, fastperiod: int = None, 
                  slowperiod: int = None, signalperiod: int = None) -> np.ndarray:
    try:
        # Ottieni parametri MACD dal config
        macd_params = config.get("trading.indicators.parameters.macd", {})
        fastperiod = fastperiod or macd_params.get("fast_period", 12)
        slowperiod = slowperiod or macd_params.get("slow_period", 26)
        signalperiod = signalperiod or macd_params.get("signal_period", 9)
        
        if len(x) < max(fastperiod, slowperiod, signalperiod):
            return np.full_like(x, np.nan)
            
        macd, _, _ = talib.MACD(x, fastperiod=fastperiod, 
                               slowperiod=slowperiod, signalperiod=signalperiod)
        return macd
    except Exception as e:
        logger.error(f"Errore MACD: {str(e)}")
        return np.full_like(x, np.nan)

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int) -> np.ndarray:
    try:
        if len(high) < timeperiod:
            return np.full_like(high, np.nan)
        return talib.ATR(high, low, close, timeperiod=timeperiod)
    except Exception as e:
        logger.error(f"Errore ATR: {str(e)}")
        return np.full_like(high, np.nan)

def calculate_stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> np.ndarray:
    try:
        if len(high) < max(fastk_period, slowk_period, slowd_period):
            return np.full_like(high, np.nan)
        slowk, _ = talib.STOCH(high, low, close, 
                              fastk_period=fastk_period,
                              slowk_period=slowk_period,
                              slowd_period=slowd_period)
        return slowk
    except Exception as e:
        logger.error(f"Errore STOCH: {str(e)}")
        return np.full_like(high, np.nan)

def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    try:
        if len(high) < timeperiod:
            return np.full_like(high, np.nan)
        return talib.ADX(high, low, close, timeperiod=timeperiod)
    except Exception as e:
        logger.error(f"Errore ADX: {str(e)}")
        return np.full_like(high, np.nan)

def calculate_close(x: np.ndarray) -> np.ndarray:
    return x

class IndicatorRegistry:
    @staticmethod
    def get_available_indicators():
        return {
            "SMA": {"func": calculate_sma, "params": ["timeperiod"]},
            "EMA": {"func": calculate_ema, "params": ["timeperiod"]},
            "RSI": {"func": calculate_rsi, "params": ["timeperiod"]},
            "MACD": {"func": calculate_macd, 
                    "params": ["fastperiod", "slowperiod", "signalperiod"]},
            "BB_UPPER": {"func": calculate_bb_upper, "params": ["timeperiod"]},
            "BB_LOWER": {"func": calculate_bb_lower, "params": ["timeperiod"]},
            "CLOSE": {"func": calculate_close, "params": []},
            "ATR": {"func": calculate_atr, "params": ["timeperiod"]},
            "STOCH": {"func": calculate_stoch, 
                     "params": ["fastk_period", "slowk_period", "slowd_period"]},
            "ADX": {"func": calculate_adx, "params": ["timeperiod"]}
        }

    @staticmethod
    def generate_random_params(param_names):
        params = {}
        ranges = config.get("trading.indicators.parameters.ranges", {})
        
        for param in param_names:
            if param == "timeperiod":
                range_config = ranges.get("timeperiod", {})
                params[param] = np.random.randint(
                    range_config.get("min", 5),
                    range_config.get("max", 50)
                )
            elif param == "fastperiod" or param == "fastk_period":
                range_config = ranges.get("fast_period", {})
                params[param] = np.random.randint(
                    range_config.get("min", 5),
                    range_config.get("max", 20)
                )
            elif param == "slowperiod" or param == "slowk_period":
                range_config = ranges.get("slow_period", {})
                params[param] = np.random.randint(
                    range_config.get("min", 15),
                    range_config.get("max", 40)
                )
            elif param == "signalperiod" or param == "slowd_period":
                range_config = ranges.get("signal_period", {})
                params[param] = np.random.randint(
                    range_config.get("min", 5),
                    range_config.get("max", 15)
                )
        return params

    @staticmethod
    def calculate_indicator(prices: np.ndarray, indicator: str, params: dict, 
                          available_indicators: dict, market_data: List[MarketData] = None) -> np.ndarray:
        try:
            indicator_info = available_indicators[indicator]
            indicator_func = indicator_info["func"]
            
            if prices is None or len(prices) < 2:
                logger.error(f"Dati prezzi insufficienti per {indicator}")
                return np.array([np.nan, np.nan])

            valid_params = {}
            for param in indicator_info["params"]:
                if param not in params:
                    valid_params[param] = IndicatorRegistry.generate_random_params([param])[param]
                else:
                    valid_params[param] = params[param]
            
            # Gestione indicatori che richiedono high/low/close
            if indicator in ["ATR", "STOCH", "ADX"]:
                if market_data is None:
                    logger.error(f"Market data richiesti per {indicator} ma non forniti")
                    return np.full_like(prices, np.nan)
                    
                high = np.array([d.high for d in market_data])
                low = np.array([d.low for d in market_data])
                close = np.array([d.close for d in market_data])
                result = indicator_func(high, low, close, **valid_params)
            else:
                result = indicator_func(prices, **valid_params)
            
            if result is None:
                logger.error(f"Indicatore {indicator} ha restituito None")
                return np.full_like(prices, np.nan)
                
            return result
            
        except Exception as e:
            logger.error(f"Errore calcolo indicatore {indicator}: {str(e)}")
            return np.full_like(prices, np.nan)
