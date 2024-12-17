import torch
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
import gc

from .common import TimeFrame, Position
from ..utils.config import config
from .genes.base import TradingGene
from .simulator_device import SimulatorDevice
from .simulator_data import MarketDataManager
from .simulator_metrics import MetricsCalculator
from .simulator_processor import SimulationProcessor

logger = logging.getLogger(__name__)

class TradingSimulator:
    def __init__(self):
        # Configurazione base
        self.initial_capital = config.get("simulator.initial_capital", 10000)
        self.min_candles = config.get("simulator.min_candles", 50)
        
        # Inizializza componenti
        self.device_manager = SimulatorDevice(config)
        self.data_manager = MarketDataManager(self.device_manager)
        self.metrics_calculator = MetricsCalculator(self.initial_capital)
        self.processor = SimulationProcessor(self.device_manager, config)
        
        # Stati
        self.positions: List[Position] = []
        self.equity_curve: List[float] = []
        self.metrics: Optional[Dict[str, Any]] = None

    def add_market_data(self, timeframe: TimeFrame, data: pd.DataFrame) -> None:
        """Aggiunge i dati di mercato al simulatore"""
        self.data_manager.add_market_data(timeframe, data)

    def run_simulation_vectorized(self, entry_conditions: torch.Tensor, gene: Optional[TradingGene] = None) -> Dict[str, Any]:
        """
        Esegue simulazione vettorizzata
        
        Args:
            entry_conditions: Condizioni di entrata
            gene: Gene opzionale per tracciamento tipo
            
        Returns:
            Dict con metriche di performance
        """
        try:
            self._reset_simulation()
            
            # Converti input in tensor
            prices = self.device_manager.to_tensor(self.data_manager.get_market_data().close)
            entry_conditions = self.device_manager.to_tensor(entry_conditions, dtype=torch.bool)
            
            # Esegui simulazione
            simulation_results = self.processor.run_simulation(
                prices=prices,
                entry_conditions=entry_conditions,
                initial_capital=self.initial_capital
            )
            
            # Calcola metriche
            metrics = self.metrics_calculator.calculate_metrics(
                pnl=simulation_results["pnl"],
                equity=simulation_results["equity"]
            )
            
            # Aggiungi gene_type se disponibile
            if gene is not None:
                metrics["gene_type"] = gene.gene_type
            
            self.metrics = metrics
            return metrics
                
        except Exception as e:
            logger.error("Error in vectorized simulation:")
            logger.error(str(e))
            raise

    def _reset_simulation(self) -> None:
        """Reset simulazione con gestione sicura della memoria"""
        try:
            # Reset delle strutture dati
            self.positions = []
            self.equity_curve = []
            self.metrics = None
            
            # Gestione memoria GPU
            if self.device_manager.use_gpu:
                try:
                    if self.device_manager.gpu_backend == "arc":
                        torch.xpu.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Error clearing GPU cache: {str(e)}")
                    
            # Garbage collection condizionale
            if self.device_manager.memory_config["periodic_gc"]:
                try:
                    gc.collect()
                    # Forza pulizia memoria GPU dopo garbage collection
                    if self.device_manager.use_gpu:
                        if self.device_manager.gpu_backend == "arc":
                            torch.xpu.empty_cache()
                        else:
                            torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Error in garbage collection: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error in simulation reset: {str(e)}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Restituisce le metriche di performance correnti"""
        return self.metrics_calculator.get_performance_metrics(self.metrics)

    @property
    def market_state(self):
        """Restituisce lo stato del mercato corrente"""
        return self.data_manager.get_market_data()

    @property
    def indicators_cache(self):
        """Restituisce la cache degli indicatori"""
        return self.data_manager.get_all_indicators()
