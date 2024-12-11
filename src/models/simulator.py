import torch
import torch.nn as nn
import torch.cuda
import numpy as np
import pandas as pd
from datetime import datetime
import talib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, ContextManager
from contextlib import nullcontext
import gc
import traceback

from .common import Signal, SignalType, MarketData, TimeFrame, Position
from ..utils.config import config
from .genes.base import TradingGene

logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

class TradingSimulator:
    def __init__(self):
        # Configurazione CUDA
        self.use_gpu = config.get("genetic.optimizer.use_gpu", False)
        self.initial_capital = config.get("simulator.initial_capital", 10000)
        self.min_candles = config.get("simulator.min_candles", 50)
        self.indicators_cache: Dict[str, torch.Tensor] = {}
        self.market_state: Optional[MarketState] = None
        self.positions: List[Position] = []
        self.equity_curve: List[float] = []
        self.metrics: Optional[Dict[str, Any]] = None
        
        # Imposta device predefinito
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.set_device(0)  # Usa prima GPU come default
            self.device = torch.device("cuda")
            self.num_gpus = torch.cuda.device_count()
            logger.info(f"Using {self.num_gpus} CUDA devices")
            
            # Configura precisione
            self.precision = config.get("genetic.optimizer.device_config.precision", "float32")
            self.dtype = torch.float16 if self.precision == "float16" else torch.float32
            
            # Configura CUDA graphs
            self.use_cuda_graphs = config.get("genetic.optimizer.device_config.cuda_graph", False)
            
            # Configura memoria
            self.memory_reserve = config.get("genetic.optimizer.device_config.memory_reserve", 2048)
            self.max_batch_size = config.get("genetic.optimizer.device_config.max_batch_size", 1024)
            
            # Abilita mixed precision se richiesto
            self.mixed_precision = config.get("genetic.optimizer.device_config.mixed_precision", False)
            if self.mixed_precision:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.max_batch_size = config.get("genetic.batch_size", 32)
            logger.info("Using CPU device")
            self.mixed_precision = False
            self.use_cuda_graphs = False
        
        self.use_gpu = config.get("genetic.optimizer.use_gpu", False)
        self.initial_capital = config.get("simulator.initial_capital", 10000)
        self.min_candles = config.get("simulator.min_candles", 50)
        self.indicators_cache: Dict[str, torch.Tensor] = {}
        self.market_state: Optional[MarketState] = None
        self.positions: List[Position] = []
        self.equity_curve: List[float] = []
        self.metrics: Optional[Dict[str, Any]] = None
        
        # Initialize CUDA related attributes
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU as default
            self.device = torch.device("cuda")
            self.num_gpus = torch.cuda.device_count()
            logger.info(f"Using {self.num_gpus} CUDA devices")
            
            # Configure precision
            self.precision = config.get("genetic.optimizer.device_config.precision", "float32")
            self.dtype = torch.float16 if self.precision == "float16" else torch.float32
            
            # Configure CUDA graphs
            self.use_cuda_graphs = config.get("genetic.optimizer.device_config.cuda_graph", False)
            self._simulation_graph = None
            self._static_inputs = None
            
            # Configure memory
            self.memory_reserve = config.get("genetic.optimizer.device_config.memory_reserve", 2048)
            self.max_batch_size = config.get("genetic.optimizer.device_config.max_batch_size", 1024)
            
            # Enable mixed precision if requested
            self.mixed_precision = config.get("genetic.optimizer.device_config.mixed_precision", False)
            if self.mixed_precision:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.max_batch_size = config.get("genetic.batch_size", 32)
            logger.info("Using CPU device")
            self.mixed_precision = False
            self.use_cuda_graphs = False
            self._simulation_graph = None

    def _to_tensor(self, data: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Converte dati in tensor PyTorch"""
        try:
            # Se non è specificato il dtype, usa quello di default
            if dtype is None:
                dtype = self.dtype
                
            # Converti numpy array in tensor
            if isinstance(data, np.ndarray):
                if data.dtype == bool:
                    tensor = torch.from_numpy(data.astype(np.uint8))
                else:
                    tensor = torch.from_numpy(data)
            elif isinstance(data, torch.Tensor):
                tensor = data
            else:
                tensor = torch.tensor(data)
            
            # Sposta su GPU e imposta dtype
            tensor = tensor.to(device=self.device, dtype=dtype)
            
            # Riconverti in bool se necessario
            if data.dtype == bool:
                tensor = tensor.bool()
                
            return tensor
        except Exception as e:
            logger.error(f"Error converting to tensor: {str(e)}")
            # Ritorna un tensor vuoto in caso di errore
            return torch.tensor([], device=self.device, dtype=dtype)

    def _to_device(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Sposta tensor su device specifico"""
        return tensor.to(device=device, dtype=self.dtype)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Converte tensor in numpy array"""
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.cpu().numpy()

    def add_market_data(self, timeframe: TimeFrame, data: pd.DataFrame) -> None:
        """Converte dati in arrays numpy per processamento veloce"""
        # Converti esplicitamente in float64 per talib
        self.market_state = MarketState(
            timestamp=data.timestamp.values,
            open=data.open.values.astype(np.float64),  # Esplicito float64
            high=data.high.values.astype(np.float64),
            low=data.low.values.astype(np.float64),
            close=data.close.values.astype(np.float64),
            volume=data.volume.values.astype(np.float64)
        )
        self._precalculate_indicators()

    def _precalculate_indicators(self) -> None:
        """Precalcola indicatori con supporto CUDA"""
        logger.info("Precalculating common indicators...")
        
        try:
            # Usa direttamente i dati in float64
            close_np = self.market_state.close
            high_np = self.market_state.high
            low_np = self.market_state.low
            
            # Aggiungi CLOSE come indicatore base
            self.indicators_cache["CLOSE"] = self._to_tensor(close_np)
            
            # Usa gli stessi periodi definiti in TradingGene.VALID_PERIODS
            periods = TradingGene.VALID_PERIODS
            
            for period in periods:
                sma = talib.SMA(close_np, timeperiod=period)
                ema = talib.EMA(close_np, timeperiod=period)
                rsi = talib.RSI(close_np, timeperiod=period)
                
                upper, middle, lower = talib.BBANDS(
                    close_np,
                    timeperiod=period
                )
                
                self.indicators_cache[f"SMA_{period}"] = self._to_tensor(sma)
                self.indicators_cache[f"EMA_{period}"] = self._to_tensor(ema) 
                self.indicators_cache[f"RSI_{period}"] = self._to_tensor(rsi)
                self.indicators_cache[f"BB_UPPER_{period}"] = self._to_tensor(upper)
                self.indicators_cache[f"BB_MIDDLE_{period}"] = self._to_tensor(middle)
                self.indicators_cache[f"BB_LOWER_{period}"] = self._to_tensor(lower)
                
        except Exception as e:
            logger.error(f"Error precalculating indicators: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _process_batch(self, batch_slice: slice, prices: torch.Tensor,
                      entry_conditions: torch.Tensor, position_active: torch.Tensor,
                      entry_prices: torch.Tensor, pnl: torch.Tensor,
                      equity: torch.Tensor, position_size_pct: float,
                      stop_loss_pct: float, take_profit_pct: float) -> None:
        """Processa un batch di dati con supporto mixed precision"""
        try:
            with torch.cuda.amp.autocast() if self.mixed_precision else nullcontext():
                current_prices = prices[batch_slice]
                current_entry_conditions = entry_conditions[batch_slice]
                prev_position_active = position_active[batch_slice.start-1]
                
                # Calcola nuove entrate
                new_entries = current_entry_conditions & ~prev_position_active
                if new_entries.any():
                    entry_idx = torch.nonzero(new_entries).squeeze()
                    position_active[batch_slice][entry_idx] = True
                    entry_prices[batch_slice][entry_idx] = current_prices[entry_idx]
                
                # Calcola uscite
                if prev_position_active:
                    entry_price = entry_prices[batch_slice.start-1]
                    price_changes = (current_prices - entry_price) / entry_price
                    
                    # Condizioni di uscita
                    stop_loss_exits = price_changes <= -stop_loss_pct
                    take_profit_exits = price_changes >= take_profit_pct
                    new_signal_exits = current_entry_conditions
                    
                    exits = stop_loss_exits | take_profit_exits | new_signal_exits
                    
                    if exits.any():
                        exit_idx = torch.nonzero(exits).squeeze()
                        position_active[batch_slice][exit_idx] = False
                        pnl[batch_slice][exit_idx] = price_changes[exit_idx] * position_size_pct * equity[batch_slice.start-1]
                
                # Aggiorna equity
                equity[batch_slice] = equity[batch_slice.start-1] + pnl[batch_slice]
                
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _reset_simulation(self) -> None:
        """Reset simulation state"""
        self.positions = []
        self.equity_curve = []
        self.metrics = None
        
        # Pulizia memoria GPU
        if self.use_gpu:
            torch.cuda.empty_cache()
            gc.collect()

    def _initialize_cuda_graph(self, entry_conditions: torch.Tensor) -> None:
        """Initialize CUDA graph for simulation if not already initialized"""
        if not self.use_cuda_graphs or not self.use_gpu:
            return
            
        try:
            if self._simulation_graph is None:
                # Create static inputs for the graph
                self._static_inputs = {
                    'prices': self._to_tensor(self.market_state.close),
                    'position_active': torch.zeros_like(entry_conditions, dtype=torch.bool, device=self.device),
                    'entry_prices': torch.zeros_like(self._to_tensor(self.market_state.close), device=self.device),
                    'pnl': torch.zeros_like(self._to_tensor(self.market_state.close), device=self.device),
                    'equity': torch.ones_like(self._to_tensor(self.market_state.close), device=self.device) * self.initial_capital
                }
                
                # Warm up run to initialize graph
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    self._simulation_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self._simulation_graph):
                        self._process_batch(
                            slice(1, len(entry_conditions)), 
                            self._static_inputs['prices'],
                            entry_conditions,
                            self._static_inputs['position_active'],
                            self._static_inputs['entry_prices'],
                            self._static_inputs['pnl'],
                            self._static_inputs['equity'],
                            config.get("trading.position.size_pct", 5) / 100,
                            config.get("trading.position.stop_loss_pct", 2) / 100,
                            config.get("trading.position.take_profit_pct", 4) / 100
                        )
                torch.cuda.current_stream().wait_stream(s)
                
        except Exception as e:
            logger.error(f"Error initializing CUDA graph: {str(e)}")
            logger.error(traceback.format_exc())
            self.use_cuda_graphs = False
            self._simulation_graph = None

    def run_simulation_vectorized(self, entry_conditions: np.ndarray) -> Dict[str, Any]:
        """Execute vectorized simulation with CUDA support"""
        self._reset_simulation()
        
        try:
            # Convert inputs to tensors
            prices = self._to_tensor(self.market_state.close)
            entry_conditions = self._to_tensor(entry_conditions, dtype=torch.bool)
            
            # Initialize or update CUDA graph if needed
            if self.use_cuda_graphs and self.use_gpu:
                self._initialize_cuda_graph(entry_conditions)
            
            # Trading parameters
            position_size_pct = config.get("trading.position.size_pct", 5) / 100
            stop_loss_pct = config.get("trading.position.stop_loss_pct", 2) / 100
            take_profit_pct = config.get("trading.position.take_profit_pct", 4) / 100
            
            # Initialize tensors on device
            position_active = torch.zeros_like(prices, dtype=torch.bool, device=self.device)
            entry_prices = torch.zeros_like(prices, device=self.device)
            pnl = torch.zeros_like(prices, device=self.device)
            equity = torch.ones_like(prices, device=self.device) * self.initial_capital
            trade_results = []
            
            # Process in batches to optimize GPU memory
            batch_size = self.max_batch_size if self.use_gpu else len(prices)
            
            for i in range(1, len(prices), batch_size):
                end_idx = min(i + batch_size, len(prices))
                batch_slice = slice(i, end_idx)
                
                # Batch processing with CUDA graph if enabled
                if self.use_cuda_graphs and self.use_gpu and self._simulation_graph is not None:
                    # Update static inputs
                    self._static_inputs['prices'].copy_(prices)
                    self._static_inputs['position_active'].copy_(position_active)
                    self._static_inputs['entry_prices'].copy_(entry_prices)
                    self._static_inputs['pnl'].copy_(pnl)
                    self._static_inputs['equity'].copy_(equity)
                    
                    # Replay graph
                    self._simulation_graph.replay()
                else:
                    self._process_batch(
                        batch_slice, prices, entry_conditions,
                        position_active, entry_prices, pnl, equity,
                        position_size_pct, stop_loss_pct, take_profit_pct
                    )
                
                # Collect batch results
                batch_pnl = pnl[batch_slice]
                trade_results.extend(batch_pnl[batch_pnl != 0].cpu().tolist())
                
                # GPU memory management
                if self.use_gpu:
                    torch.cuda.empty_cache()
            
            # Calculate final metrics
            total_trades = len(trade_results)
            
            if total_trades == 0:
                self.metrics = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "final_capital": float(self.initial_capital),
                    "max_drawdown": 0,
                    "sharpe_ratio": 0,
                    "profit_factor": 0
                }
            else:
                # Rest of the metrics calculation remains the same...
                winning_trades = sum(1 for x in trade_results if x > 0)
                equity_np = equity.cpu().numpy()
                peaks = np.maximum.accumulate(equity_np)
                drawdowns = (peaks - equity_np) / peaks
                max_drawdown = float(np.max(drawdowns))
                
                returns = np.diff(equity_np) / equity_np[:-1]
                returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
                
                if len(returns) > 0 and np.std(returns) > 0:
                    sharpe = float(np.sqrt(252) * (np.mean(returns) / np.std(returns)))
                else:
                    sharpe = 0
                
                gross_profits = sum(x for x in trade_results if x > 0)
                gross_losses = abs(sum(x for x in trade_results if x < 0))
                profit_factor = float(gross_profits / gross_losses if gross_losses != 0 else 0)
                
                self.metrics = {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "win_rate": winning_trades / total_trades,
                    "total_pnl": float(equity[-1] - self.initial_capital),
                    "final_capital": float(equity[-1]),
                    "max_drawdown": max_drawdown,
                    "sharpe_ratio": sharpe,
                    "profit_factor": profit_factor
                }
            
            return self.metrics
            
        except Exception as e:
            logger.error("Error in vectorized simulation:")
            logger.error(traceback.format_exc())
            raise

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Returns current performance metrics"""
        if self.metrics is None:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "final_capital": self.initial_capital,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "profit_factor": 0
            }
        return self.metrics
