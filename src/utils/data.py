# src/utils/data.py
import pandas as pd
import numpy as np
from typing import Dict

def prepare_market_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    required_columns = {
        'timestamp': ['timestamp', 'time', 'datetime', 'date'],
        'open': ['open', 'Open'],
        'high': ['high', 'High'],
        'low': ['low', 'Low'],
        'close': ['close', 'Close'],
        'volume': ['volume', 'Volume']
    }
    
    for required, alternatives in required_columns.items():
        if required not in df.columns:
            found = False
            for alt in alternatives:
                if alt in df.columns:
                    df = df.rename(columns={alt: required})
                    found = True
                    break
            if not found:
                raise ValueError(f"Colonna {required} non trovata nel dataset")

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values('timestamp').drop_duplicates('timestamp')
    
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if df.isnull().any().any():
        df = df.ffill()
    
    return df

def resample_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df.copy()
    df.set_index('timestamp', inplace=True)
    
    timeframe_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '1h': '1h',
        '4h': '4h',
        '1d': '1D'
    }
    
    resample_tf = timeframe_map.get(timeframe, timeframe)
    
    resampled = df.resample(resample_tf).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    return resampled.reset_index()

def load_and_prepare_data(file_path: str) -> Dict[str, pd.DataFrame]:
    base_data = prepare_market_data(file_path)
    
    return {
        '1m': base_data,
        '5m': resample_timeframe(base_data, '5m'),
        '15m': resample_timeframe(base_data, '15m'),
        '1h': resample_timeframe(base_data, '1h'),
        '4h': resample_timeframe(base_data, '4h'),
        '1d': resample_timeframe(base_data, '1d')
    }