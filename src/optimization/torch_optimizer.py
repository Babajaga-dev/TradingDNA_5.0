import torch
import logging
from typing import List, Dict, Any

from ..models.simulator import TradingSimulator
from ..utils.config import config
from .torch_device import TorchDeviceManager
from .torch_data import TorchDataManager
from .torch_evaluator import TorchEvaluator
from .torch_simulator import TorchSimulator

logger = logging.getLogger(__name__)

class TorchOptimizer:
    """Ottimizzatore con supporto PyTorch"""
    
    def __init__(self, simulator: TradingSimulator):
        """
        Args:
            simulator: Simulatore di trading
        """
        # Inizializza managers
        self.device_manager = TorchDeviceManager()
        self.device = self.device_manager.get_best_device(config)
        self.device_manager.setup_device(self.device, config)
        
        self.data_manager = TorchDataManager(self.device, config)
        self.simulator_torch = TorchSimulator(config)
        self.evaluator = TorchEvaluator(simulator, self.data_manager, config)
        
        # Riferimento al simulatore originale
        self.simulator = simulator
        
        logger.info(f"Initialized TorchOptimizer with device: {self.device}")
        logger.info(f"Precision: {self.data_manager.precision}")
        logger.info(f"Using batch normalization: {self.data_manager.batch_norm}")

    def prepare_data(self, market_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepara i dati per l'ottimizzazione
        
        Args:
            market_data: Dati di mercato
            
        Returns:
            Dati convertiti in tensori PyTorch
        """
        return self.data_manager.prepare_data(market_data)

    def evaluate_population(self, population: List, market_data: Dict[str, torch.Tensor]) -> List[float]:
        """
        Valuta una popolazione
        
        Args:
            population: Lista di geni da valutare
            market_data: Dati di mercato come tensori
            
        Returns:
            Lista di fitness scores
        """
        return self.evaluator.evaluate_population_parallel(population, market_data)

    def run_simulation(self, 
                      entry_conditions: torch.Tensor,
                      prices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Esegue una simulazione
        
        Args:
            entry_conditions: Condizioni di entrata
            prices: Prezzi di mercato
            
        Returns:
            Risultati della simulazione
        """
        return self.simulator_torch.run_simulation_vectorized(
            entry_conditions=entry_conditions,
            prices=prices,
            device=self.device,
            initial_capital=self.simulator.initial_capital
        )

def add_torch_simulation(simulator_class: type) -> None:
    """
    Aggiunge metodi per simulazione PyTorch al TradingSimulator
    
    Args:
        simulator_class: Classe TradingSimulator
    """
    def run_simulation_vectorized_torch(self: TradingSimulator, 
                                      entry_conditions: torch.Tensor,
                                      device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Versione PyTorch della simulazione vettorizzata
        
        Args:
            entry_conditions: Condizioni di entrata
            device: Device PyTorch
            
        Returns:
            Dizionario delle metriche
        """
        try:
            # Crea simulatore PyTorch
            simulator = TorchSimulator(config)
            
            # Esegui simulazione
            return simulator.run_simulation_vectorized(
                entry_conditions=entry_conditions,
                prices=torch.tensor(self.market_state.close, device=device),
                device=device,
                initial_capital=self.initial_capital
            )
            
        except Exception as e:
            logger.error("Error in vectorized simulation:")
            logger.error(str(e))
            return simulator._get_empty_metrics(device, self.initial_capital)

    # Aggiungi metodo al TradingSimulator
    simulator_class.run_simulation_vectorized_torch = run_simulation_vectorized_torch
