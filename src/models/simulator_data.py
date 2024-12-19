import torch
import pandas as pd
import talib
import logging
from dataclasses import dataclass
from typing import Dict, Optional
import traceback
import os

from .common import TimeFrame
from .genes.base import TradingGene
from .genes.indicators import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_bb_upper, calculate_bb_lower, to_numpy
)
from ..utils.config import config

logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    timestamp: torch.Tensor
    open: torch.Tensor
    high: torch.Tensor
    low: torch.Tensor
    close: torch.Tensor
    volume: torch.Tensor

class MarketDataManager:
    def __init__(self, device_manager):
        self.device_manager = device_manager
        self.market_state: Optional[MarketState] = None
        self.indicators_cache: Dict[str, torch.Tensor] = {}
        
        # Carica dati dal file se specificato
        data_file = config.get("simulator.data_file")
        if data_file:
            self._load_from_file(data_file)

    def _load_from_file(self, filename: str) -> None:
        """
        Carica i dati di mercato dal file specificato
        
        Args:
            filename: Nome del file da caricare
        """
        try:
            # Aggiungi estensione .csv se non presente
            if not filename.lower().endswith('.csv'):
                filename = f"{filename}.csv"
            
            # Costruisci il path completo
            data_dir = "data"
            file_path = os.path.join(data_dir, filename)
            
            if not os.path.exists(file_path):
                logger.error(f"File non trovato: {file_path}")
                return
                
            logger.info(f"Caricamento dati da {file_path}")
            
            # Carica il file CSV
            data = pd.read_csv(file_path)
            
            # Verifica e rinomina colonne se necessario
            column_mapping = {
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            
            # Rinomina colonne se hanno nomi diversi
            for std_name, possible_names in {
                'timestamp': ['timestamp', 'time', 'date'],
                'open': ['open', 'Open'],
                'high': ['high', 'High'],
                'low': ['low', 'Low'],
                'close': ['close', 'Close'],
                'volume': ['volume', 'Volume']
            }.items():
                for col_name in possible_names:
                    if col_name in data.columns:
                        data = data.rename(columns={col_name: std_name})
                        break
            
            # Verifica colonne necessarie
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                raise ValueError(f"Colonne mancanti nel file: {missing}")
            
            # Converti timestamp se necessario
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Ordina per timestamp
            data = data.sort_values('timestamp')
            
            # Aggiungi i dati
            self.add_market_data(TimeFrame.M1, data)
            
            logger.info(f"Caricati {len(data)} candles da {filename}")
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del file {filename}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def add_market_data(self, timeframe: TimeFrame, data: pd.DataFrame) -> None:
        """
        Aggiunge i dati di mercato e precalcola gli indicatori tecnici.
        
        Args:
            timeframe: TimeFrame - Il timeframe dei dati (es. M1, M5, etc.)
            data: pd.DataFrame - DataFrame contenente i dati di mercato
        """
        try:
            logger.info(f"Adding market data for timeframe {timeframe.value}")
            
            # Verifica colonne necessarie
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                raise ValueError(f"Missing required columns in market data: {missing}")
            
            # Converti direttamente in tensori torch
            device = self.device_manager.device
            dtype = torch.float32

            self.market_state = MarketState(
                timestamp=torch.tensor(data['timestamp'].astype('int64').astype(int).values, device=device),
                open=torch.tensor(data['open'].values, dtype=dtype, device=device),
                high=torch.tensor(data['high'].values, dtype=dtype, device=device),
                low=torch.tensor(data['low'].values, dtype=dtype, device=device),
                close=torch.tensor(data['close'].values, dtype=dtype, device=device),
                volume=torch.tensor(data['volume'].values, dtype=dtype, device=device)
            )
            
            # Precalcola indicatori
            self._precalculate_indicators()
            
            logger.info(f"Successfully added {len(data)} candles of market data")
            
        except Exception as e:
            logger.error(f"Error adding market data: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _precalculate_indicators(self) -> None:
        """Precalcola tutti gli indicatori tecnici necessari"""
        try:
            # Aggiungi prezzo di chiusura alla cache
            self.indicators_cache["CLOSE"] = self.market_state.close
            
            # Calcola indicatori per vari periodi
            for period in TradingGene.VALID_PERIODS:
                # SMA
                sma = calculate_sma(self.market_state.close, period)
                self.indicators_cache[f"SMA_{period}"] = sma
                
                # EMA
                ema = calculate_ema(self.market_state.close, period)
                self.indicators_cache[f"EMA_{period}"] = ema
                
                # RSI
                rsi = calculate_rsi(self.market_state.close, period)
                self.indicators_cache[f"RSI_{period}"] = rsi
                
                # Bollinger Bands
                upper = calculate_bb_upper(self.market_state.close, period)
                lower = calculate_bb_lower(self.market_state.close, period)
                self.indicators_cache[f"BB_UPPER_{period}"] = upper
                self.indicators_cache[f"BB_LOWER_{period}"] = lower
            
            logger.info("Successfully precalculated technical indicators")
            logger.debug(f"Available indicators: {list(self.indicators_cache.keys())}")
            
        except Exception as e:
            logger.error(f"Error precalculating indicators: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_market_data(self) -> MarketState:
        """Restituisce i dati di mercato correnti"""
        if self.market_state is None:
            raise ValueError("No market data available. Call add_market_data first.")
        return self.market_state

    def get_indicator(self, name: str) -> Optional[torch.Tensor]:
        """Restituisce un indicatore specifico dalla cache"""
        return self.indicators_cache.get(name)

    def get_all_indicators(self) -> Dict[str, torch.Tensor]:
        """Restituisce tutti gli indicatori precalcolati"""
        return self.indicators_cache

    def clear_cache(self) -> None:
        """Pulisce la cache degli indicatori"""
        self.indicators_cache.clear()
        if self.device_manager.use_gpu:
            if self.device_manager.gpu_backend == "arc":
                torch.xpu.empty_cache()
            else:
                torch.cuda.empty_cache()
