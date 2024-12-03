import pandas as pd
import numpy as np
from typing import Dict

def prepare_market_data(file_path: str) -> pd.DataFrame:
    """
    Prepara i dati di mercato nel formato corretto per il sistema
    """
    # Carica i dati
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

    # Converti timestamp in datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ordina per timestamp
    df = df.sort_values('timestamp')
    
    # Rimuovi eventuali duplicati
    df = df.drop_duplicates('timestamp')
    
    # Converti le colonne numeriche
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Gestisci valori nulli
    if df.isnull().any().any():
        print("Attenzione: Trovati valori nulli nel dataset")
        df = df.ffill()
    
    return df

def resample_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Ricampiona i dati a un timeframe superiore
    """
    df = df.copy()
    df.set_index('timestamp', inplace=True)
    
    # Mappa dei timeframe per la conversione
    timeframe_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D'
    }
    
    # Usa il timeframe mappato per il resampling
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
    """
    Carica e prepara i dati per tutti i timeframe necessari.
    Usa i timeframe nel formato corretto per il sistema di trading (1m, 5m, etc.)
    """
    # Carica e prepara i dati base
    base_data = prepare_market_data(file_path)
    
    # Prepara i dati per ogni timeframe usando i formati del sistema
    timeframes = {
        '1m': base_data,  # dati al minuto originali
        '5m': resample_timeframe(base_data, '5m'),
        '15m': resample_timeframe(base_data, '15m'),
        '1h': resample_timeframe(base_data, '1h'),
        '4h': resample_timeframe(base_data, '4h'),
        '1d': resample_timeframe(base_data, '1d')
    }
    
    return timeframes
   