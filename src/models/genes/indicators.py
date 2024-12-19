import torch
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

def to_numpy(x: torch.Tensor) -> torch.Tensor:
    """Converte un tensor in numpy array per talib"""
    if x.device.type in ['cuda', 'xpu']:
        return x.cpu().numpy()
    return x.numpy()

def from_numpy(x: torch.Tensor) -> torch.Tensor:
    """Converte un numpy array in tensor mantenendo il device"""
    tensor = torch.from_numpy(x)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return tensor.xpu()
    elif torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def calculate_bb_upper(x: torch.Tensor, timeperiod: int) -> torch.Tensor:
    try:
        if len(x) < timeperiod:
            return torch.full_like(x, float('nan'))
            
        # Calcola SMA
        kernel = torch.ones(1, 1, timeperiod, device=x.device) / timeperiod
        x_padded = torch.nn.functional.pad(x.view(1, 1, -1), (timeperiod-1, 0))
        sma = torch.nn.functional.conv1d(x_padded, kernel).squeeze()
        
        # Calcola deviazione standard
        x_squared = x * x
        x_squared_padded = torch.nn.functional.pad(x_squared.view(1, 1, -1), (timeperiod-1, 0))
        squared_mean = torch.nn.functional.conv1d(x_squared_padded, kernel).squeeze()
        std = torch.sqrt(squared_mean - sma * sma)
        
        # Calcola banda superiore (SMA + 2*STD)
        upper = sma + 2 * std
        
        # Padding per mantenere la lunghezza originale
        padding = torch.full((timeperiod-1,), float('nan'), device=x.device)
        return torch.cat([padding, upper])
    except Exception as e:
        logger.error(f"Errore BB_UPPER: {str(e)}")
        return torch.full_like(x, float('nan'))

def calculate_bb_lower(x: torch.Tensor, timeperiod: int) -> torch.Tensor:
    try:
        if len(x) < timeperiod:
            return torch.full_like(x, float('nan'))
            
        # Calcola SMA
        kernel = torch.ones(1, 1, timeperiod, device=x.device) / timeperiod
        x_padded = torch.nn.functional.pad(x.view(1, 1, -1), (timeperiod-1, 0))
        sma = torch.nn.functional.conv1d(x_padded, kernel).squeeze()
        
        # Calcola deviazione standard
        x_squared = x * x
        x_squared_padded = torch.nn.functional.pad(x_squared.view(1, 1, -1), (timeperiod-1, 0))
        squared_mean = torch.nn.functional.conv1d(x_squared_padded, kernel).squeeze()
        std = torch.sqrt(squared_mean - sma * sma)
        
        # Calcola banda inferiore (SMA - 2*STD)
        lower = sma - 2 * std
        
        # Padding per mantenere la lunghezza originale
        padding = torch.full((timeperiod-1,), float('nan'), device=x.device)
        return torch.cat([padding, lower])
    except Exception as e:
        logger.error(f"Errore BB_LOWER: {str(e)}")
        return torch.full_like(x, float('nan'))

def calculate_sma(x: torch.Tensor, timeperiod: int) -> torch.Tensor:
    try:
        if len(x) < timeperiod:
            return torch.full_like(x, float('nan'))
            
        # Usa conv1d per calcolare la media mobile
        kernel = torch.ones(1, 1, timeperiod, device=x.device) / timeperiod
        x_padded = torch.nn.functional.pad(x.view(1, 1, -1), (timeperiod-1, 0))
        sma = torch.nn.functional.conv1d(x_padded, kernel).squeeze()
        
        # Padding per mantenere la lunghezza originale
        padding = torch.full((timeperiod-1,), float('nan'), device=x.device)
        return torch.cat([padding, sma])
    except Exception as e:
        logger.error(f"Errore SMA: {str(e)}")
        return torch.full_like(x, float('nan'))

def calculate_ema(x: torch.Tensor, timeperiod: int) -> torch.Tensor:
    try:
        if len(x) < timeperiod:
            return torch.full_like(x, float('nan'))
            
        alpha_multiplier = config.get("trading.indicators.parameters.ema.alpha_multiplier", 2.0)
        alpha = alpha_multiplier / (timeperiod + 1)
        
        # Calcola SMA iniziale
        kernel = torch.ones(1, 1, timeperiod, device=x.device) / timeperiod
        x_padded = torch.nn.functional.pad(x.view(1, 1, -1), (timeperiod-1, 0))
        sma = torch.nn.functional.conv1d(x_padded, kernel).squeeze()
        
        # Prepara il vettore dei pesi per EMA
        weights = torch.zeros(len(x), device=x.device)
        weights[timeperiod-1:] = alpha * (1 - alpha) ** torch.arange(len(x) - timeperiod + 1, device=x.device)
        
        # Calcola EMA usando convoluzione
        x_weighted = x * weights
        ema = torch.zeros_like(x)
        ema[:timeperiod-1] = float('nan')
        ema[timeperiod-1] = sma[timeperiod-1]
        ema[timeperiod:] = x_weighted[timeperiod:].cumsum(0) / weights[timeperiod:].cumsum(0)
        
        return ema
    except Exception as e:
        logger.error(f"Errore EMA: {str(e)}")
        return torch.full_like(x, float('nan'))

def calculate_rsi(x: torch.Tensor, timeperiod: int) -> torch.Tensor:
    try:
        if len(x) < timeperiod + 1:
            return torch.full_like(x, float('nan'))
        
        # Ottieni parametri RSI dal config
        rsi_params = config.get("trading.indicators.parameters.rsi", {})
        epsilon = float(rsi_params.get("epsilon", 1e-10))  # Converti in float
        scale = float(rsi_params.get("scale", 100.0))     # Converti in float
        
        # Calcola RSI usando operazioni torch
        delta = x[1:] - x[:-1]
        gains = torch.where(delta > 0, delta, torch.zeros_like(delta))
        losses = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        # Usa torch.conv1d per la media mobile
        kernel_size = timeperiod
        kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
        
        # Reshape per conv1d
        gains_padded = torch.nn.functional.pad(gains.view(1, 1, -1), (kernel_size-1, 0))
        losses_padded = torch.nn.functional.pad(losses.view(1, 1, -1), (kernel_size-1, 0))
        
        avg_gains = torch.nn.functional.conv1d(gains_padded, kernel).squeeze()
        avg_losses = torch.nn.functional.conv1d(losses_padded, kernel).squeeze()
        
        # Aggiungi epsilon come tensor
        epsilon_tensor = torch.tensor(epsilon, device=x.device)
        rs = avg_gains / (avg_losses + epsilon_tensor)
        
        # Calcola RSI usando tensori
        scale_tensor = torch.tensor(scale, device=x.device)
        rsi = scale_tensor - (scale_tensor / (1 + rs))
        
        # Padding per mantenere la lunghezza originale
        padding = torch.full((timeperiod,), float('nan'), device=x.device)
        return torch.cat([padding, rsi])
    except Exception as e:
        logger.error(f"Errore RSI: {str(e)}")
        return torch.full_like(x, float('nan'))

def calculate_macd(x: torch.Tensor, fastperiod: int = None, 
                  slowperiod: int = None, signalperiod: int = None) -> torch.Tensor:
    try:
        # Ottieni parametri MACD dal config
        macd_params = config.get("trading.indicators.parameters.macd", {})
        fastperiod = fastperiod or macd_params.get("fast_period", 12)
        slowperiod = slowperiod or macd_params.get("slow_period", 26)
        signalperiod = signalperiod or macd_params.get("signal_period", 9)
        
        if len(x) < max(fastperiod, slowperiod, signalperiod):
            return torch.full_like(x, float('nan'))
        
        # Calcola EMA veloce e lenta
        fast_ema = calculate_ema(x, fastperiod)
        slow_ema = calculate_ema(x, slowperiod)
        
        # MACD è la differenza tra EMA veloce e lenta
        macd_line = fast_ema - slow_ema
        
        return macd_line
    except Exception as e:
        logger.error(f"Errore MACD: {str(e)}")
        return torch.full_like(x, float('nan'))

def calculate_atr(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, timeperiod: int) -> torch.Tensor:
    try:
        if len(high) < timeperiod:
            return torch.full_like(high, float('nan'))
            
        # Calcola True Range
        prev_close = torch.cat([torch.tensor([float('nan')], device=close.device), close[:-1]])
        h_l = high - low
        h_pc = torch.abs(high - prev_close)
        l_pc = torch.abs(low - prev_close)
        
        tr = torch.maximum(h_l, torch.maximum(h_pc, l_pc))
        
        # Calcola media mobile del True Range
        kernel = torch.ones(1, 1, timeperiod, device=high.device) / timeperiod
        tr_padded = torch.nn.functional.pad(tr.view(1, 1, -1), (timeperiod-1, 0))
        atr = torch.nn.functional.conv1d(tr_padded, kernel).squeeze()
        
        # Padding per mantenere la lunghezza originale
        padding = torch.full((timeperiod-1,), float('nan'), device=high.device)
        return torch.cat([padding, atr])
    except Exception as e:
        logger.error(f"Errore ATR: {str(e)}")
        return torch.full_like(high, float('nan'))

def calculate_stoch(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor,
                   fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> torch.Tensor:
    try:
        if len(high) < max(fastk_period, slowk_period, slowd_period):
            return torch.full_like(high, float('nan'))
            
        # Calcola massimi e minimi su fastk_period
        high_padded = high.unfold(0, fastk_period, 1)
        low_padded = low.unfold(0, fastk_period, 1)
        
        highest_high = torch.max(high_padded, dim=1)[0]
        lowest_low = torch.min(low_padded, dim=1)[0]
        
        # Calcola %K veloce
        k_fast = 100 * (close[fastk_period-1:] - lowest_low) / (highest_high - lowest_low + 1e-10)
        
        # Padding per mantenere la lunghezza originale
        padding = torch.full((fastk_period-1,), float('nan'), device=high.device)
        return torch.cat([padding, k_fast])
    except Exception as e:
        logger.error(f"Errore STOCH: {str(e)}")
        return torch.full_like(high, float('nan'))

def calculate_adx(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor, timeperiod: int = 14) -> torch.Tensor:
    try:
        if len(high) < timeperiod:
            return torch.full_like(high, float('nan'))
            
        # Calcola +DM e -DM
        high_diff = high[1:] - high[:-1]
        low_diff = low[:-1] - low[1:]
        
        plus_dm = torch.where((high_diff > low_diff) & (high_diff > 0), high_diff, torch.zeros_like(high_diff))
        minus_dm = torch.where((low_diff > high_diff) & (low_diff > 0), low_diff, torch.zeros_like(low_diff))
        
        # Calcola TR
        prev_close = torch.cat([torch.tensor([float('nan')], device=close.device), close[:-1]])
        h_l = high - low
        h_pc = torch.abs(high - prev_close)
        l_pc = torch.abs(low - prev_close)
        tr = torch.maximum(h_l, torch.maximum(h_pc, l_pc))
        
        # Calcola medie mobili di +DM, -DM e TR
        kernel = torch.ones(1, 1, timeperiod, device=high.device) / timeperiod
        
        plus_dm_padded = torch.nn.functional.pad(plus_dm.view(1, 1, -1), (timeperiod-1, 0))
        minus_dm_padded = torch.nn.functional.pad(minus_dm.view(1, 1, -1), (timeperiod-1, 0))
        tr_padded = torch.nn.functional.pad(tr.view(1, 1, -1), (timeperiod-1, 0))
        
        plus_dm_smooth = torch.nn.functional.conv1d(plus_dm_padded, kernel).squeeze()
        minus_dm_smooth = torch.nn.functional.conv1d(minus_dm_padded, kernel).squeeze()
        tr_smooth = torch.nn.functional.conv1d(tr_padded, kernel).squeeze()
        
        # Calcola +DI e -DI
        plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
        
        # Calcola DX e ADX
        dx = 100 * torch.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        
        # Media mobile di DX per ottenere ADX
        dx_padded = torch.nn.functional.pad(dx.view(1, 1, -1), (timeperiod-1, 0))
        adx = torch.nn.functional.conv1d(dx_padded, kernel).squeeze()
        
        # Padding per mantenere la lunghezza originale
        padding = torch.full((timeperiod*2-1,), float('nan'), device=high.device)
        return torch.cat([padding, adx])
    except Exception as e:
        logger.error(f"Errore ADX: {str(e)}")
        return torch.full_like(high, float('nan'))

def calculate_close(x: torch.Tensor) -> torch.Tensor:
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
                params[param] = torch.randint(
                    range_config.get("min", 5),
                    range_config.get("max", 50),
                    (1,)
                ).item()
            elif param == "fastperiod" or param == "fastk_period":
                range_config = ranges.get("fast_period", {})
                params[param] = torch.randint(
                    range_config.get("min", 5),
                    range_config.get("max", 20),
                    (1,)
                ).item()
            elif param == "slowperiod" or param == "slowk_period":
                range_config = ranges.get("slow_period", {})
                params[param] = torch.randint(
                    range_config.get("min", 15),
                    range_config.get("max", 40),
                    (1,)
                ).item()
            elif param == "signalperiod" or param == "slowd_period":
                range_config = ranges.get("signal_period", {})
                params[param] = torch.randint(
                    range_config.get("min", 5),
                    range_config.get("max", 15),
                    (1,)
                ).item()
        return params

    @staticmethod
    def calculate_indicator(prices: torch.Tensor, indicator: str, params: dict, 
                          available_indicators: dict, market_data: List[MarketData] = None) -> torch.Tensor:
        try:
            indicator_info = available_indicators[indicator]
            indicator_func = indicator_info["func"]
            
            if prices is None or len(prices) < 2:
                logger.error(f"Dati prezzi insufficienti per {indicator}")
                return torch.tensor([float('nan'), float('nan')], device=prices.device)

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
                    return torch.full_like(prices, float('nan'))
                    
                high = torch.tensor([d.high for d in market_data], device=prices.device)
                low = torch.tensor([d.low for d in market_data], device=prices.device)
                close = torch.tensor([d.close for d in market_data], device=prices.device)
                result = indicator_func(high, low, close, **valid_params)
            else:
                result = indicator_func(prices, **valid_params)
            
            if result is None:
                logger.error(f"Indicatore {indicator} ha restituito None")
                return torch.full_like(prices, float('nan'))
                
            return result
            
        except Exception as e:
            logger.error(f"Errore calcolo indicatore {indicator}: {str(e)}")
            return torch.full_like(prices, float('nan'))
