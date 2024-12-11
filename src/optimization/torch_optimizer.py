# src/optimization/torch_optimizer.py
import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union, cast
from dataclasses import dataclass
import os
import platform
import psutil
from pathlib import Path
from contextlib import contextmanager

from ..models.genes.base import TradingGene
from ..models.simulator import TradingSimulator
from ..utils.config import config

logger = logging.getLogger(__name__)

@dataclass
class DeviceConfig:
    """Configurazione di un dispositivo (CPU o GPU)"""
    device_type: str
    device_index: int
    name: str
    memory_total: int
    memory_free: int
    compute_capability: Optional[Tuple[int, int]]

class TorchDeviceManager:
    """Gestisce i dispositivi disponibili per PyTorch"""
    
    def __init__(self):
        self.devices: List[DeviceConfig] = []
        self._detect_devices()

    def _detect_devices(self) -> None:
        """Rileva e configura dispositivi disponibili (CPU e GPU)"""
        try:
            # Rileva memoria CPU
            vm = psutil.virtual_memory()
            
            # Sempre aggiungi CPU
            self.devices.append(DeviceConfig(
                device_type="cpu",
                device_index=-1,
                name="CPU",
                memory_total=vm.total,
                memory_free=vm.available,
                compute_capability=None
            ))

            # Controlla GPU disponibili
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        gpu = torch.cuda.get_device_properties(i)
                        mem_free, mem_total = torch.cuda.mem_get_info(i)
                        self.devices.append(DeviceConfig(
                            device_type="cuda",
                            device_index=i,
                            name=torch.cuda.get_device_name(i),
                            memory_total=mem_total,
                            memory_free=mem_free,
                            compute_capability=(gpu.major, gpu.minor)
                        ))
                    except Exception as e:
                        logger.error(f"Error detecting GPU {i}: {e}")

            # Log dispositivi rilevati
            logger.info("Detected devices:")
            for dev in self.devices:
                logger.info(f"- {dev.name} ({dev.device_type})")
                if dev.device_type == "cuda":
                    logger.info(f"  Compute capability: {dev.compute_capability}")
                    logger.info(f"  Memory: {dev.memory_total / 1024**3:.1f}GB")
                    
        except Exception as e:
            logger.error(f"Error detecting devices: {e}")
            # Fallback a CPU
            vm = psutil.virtual_memory()
            self.devices = [DeviceConfig(
                device_type="cpu",
                device_index=-1,
                name="CPU",
                memory_total=vm.total,
                memory_free=vm.available,
                compute_capability=None
            )]

    def get_best_device(self) -> torch.device:
        """
        Seleziona il miglior dispositivo disponibile
        
        Returns:
            Device PyTorch ottimale
        """
        try:
            use_gpu = config.get("genetic.optimizer.use_gpu", False)
            
            if use_gpu and len([d for d in self.devices if d.device_type == "cuda"]) > 0:
                # Seleziona GPU con più memoria libera
                best_gpu = max(
                    [d for d in self.devices if d.device_type == "cuda"],
                    key=lambda x: x.memory_free
                )
                logger.info(f"Selected GPU device: {best_gpu.name}")
                return torch.device(f"cuda:{best_gpu.device_index}")
            else:
                logger.info("Using CPU device")
                return torch.device("cpu")
                
        except Exception as e:
            logger.error(f"Error selecting device: {e}")
            return torch.device("cpu")

class TorchOptimizer:
    """Ottimizzatore con supporto PyTorch"""
    
    def __init__(self, simulator: TradingSimulator):
        """
        Args:
            simulator: Simulatore di trading
        """
        self.device_manager = TorchDeviceManager()
        self.device = self.device_manager.get_best_device()
        self.simulator = simulator
        self.precision = config.get("genetic.optimizer.precision", "float32")
        self.batch_norm = config.get("genetic.optimizer.batch_norm", True)
        
        # Set numero di thread per CPU
        if self.device.type == "cpu":
            torch_threads = config.get("genetic.optimizer.torch_threads", None)
            if torch_threads:
                torch.set_num_threads(torch_threads)
                
        logger.info(f"Initialized TorchOptimizer with device: {self.device}")
        logger.info(f"Precision: {self.precision}")
        logger.info(f"Using batch normalization: {self.batch_norm}")

    @contextmanager
    def _gpu_memory_manager(self):
        """Context manager per gestire la memoria GPU"""
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def prepare_data(self, market_data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Converte dati di mercato in tensori PyTorch
        
        Args:
            market_data: Dizionario con dati di mercato
            
        Returns:
            Dizionario con tensori PyTorch
        """
        try:
            torch_data: Dict[str, torch.Tensor] = {}
            dtype = torch.float32 if self.precision == "float32" else torch.float16

            for key, value in market_data.items():
                if isinstance(value, np.ndarray):
                    tensor = torch.tensor(value, dtype=dtype, device=self.device)
                    if self.batch_norm:
                        # Applica normalizzazione
                        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
                    torch_data[key] = tensor

            return torch_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return {}

    def evaluate_population_parallel(self, 
                                   population: List[TradingGene],
                                   market_data: Dict[str, torch.Tensor]) -> List[float]:
        """
        Valuta popolazione in parallelo su GPU
        
        Args:
            population: Lista di geni da valutare
            market_data: Dati di mercato come tensori
            
        Returns:
            Lista di fitness scores
        """
        try:
            batch_size = config.get("genetic.batch_size", 32)
            fitness_scores: List[float] = []

            with self._gpu_memory_manager():
                for i in range(0, len(population), batch_size):
                    batch = population[i:i + batch_size]
                    batch_conditions: List[torch.Tensor] = []

                    # Genera condizioni di entrata per il batch
                    for gene in batch:
                        try:
                            conditions = gene.generate_entry_conditions(market_data)
                            batch_conditions.append(conditions)
                        except Exception as e:
                            logger.error(f"Error generating conditions for gene: {e}")
                            continue

                    # Converti in tensor e processa su device
                    if batch_conditions:
                        try:
                            conditions_tensor = torch.stack(batch_conditions).to(self.device)
                            
                            # Esegui simulazione vettorizzata
                            with torch.no_grad():
                                metrics = self.simulator.run_simulation_vectorized_torch(
                                    conditions_tensor,
                                    self.device
                                )

                            # Calcola fitness scores
                            for j, gene in enumerate(batch):
                                metrics_dict = {
                                    k: v[j].item() if isinstance(v, torch.Tensor) else v
                                    for k, v in metrics.items()
                                }
                                fitness = self._calculate_fitness(metrics_dict)
                                fitness_scores.append(fitness)
                                gene.fitness_score = fitness
                                
                        except Exception as e:
                            logger.error(f"Error processing batch: {e}")
                            continue

            return fitness_scores
            
        except Exception as e:
            logger.error(f"Error evaluating population: {e}")
            return [0.0] * len(population)

    def _calculate_fitness(self, metrics: Dict[str, Any]) -> float:
        """
        Calcola fitness score ottimizzato per GPU
        
        Args:
            metrics: Dizionario delle metriche
            
        Returns:
            Score di fitness
        """
        try:
            if metrics["total_trades"] < config.get("genetic.min_trades", 50):
                return 0.0

            weights = config.get("genetic.fitness_weights", {})
            
            # Calcola componenti del fitness
            profit_score = (
                weights.get("profit_score", {}).get("total_pnl", 0.35) * 
                metrics["total_pnl"] / self.simulator.initial_capital +
                weights.get("profit_score", {}).get("max_drawdown", 0.25) * 
                (1 - metrics["max_drawdown"]) +
                weights.get("profit_score", {}).get("sharpe_ratio", 0.40) * 
                max(0, metrics["sharpe_ratio"]) / 3
            )

            quality_score = (
                weights.get("quality_score", {}).get("win_rate", 0.45) * 
                metrics["win_rate"] +
                weights.get("quality_score", {}).get("trade_frequency", 0.25) * 
                min(1.0, metrics["total_trades"] / 100)
            )

            if "profit_factor" in metrics:
                consistency = weights.get("quality_score", {}).get("consistency", 0.30) * \
                             (metrics["profit_factor"] - 1) / 2
                quality_score += consistency

            # Score finale
            final_score = (
                weights.get("final_weights", {}).get("profit", 0.50) * profit_score +
                weights.get("final_weights", {}).get("quality", 0.40) * quality_score
            )

            return max(0.0, final_score)
            
        except Exception as e:
            logger.error(f"Error calculating fitness: {e}")
            return 0.0

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
            # Converti parametri in tensori
            position_size_pct = config.get("trading.position.size_pct", 5) / 100
            stop_loss_pct = config.get("trading.position.stop_loss_pct", 2) / 100
            take_profit_pct = config.get("trading.position.take_profit_pct", 4) / 100

            # Prepara tensori su device
            prices = torch.tensor(self.market_state.close, device=device)
            position_active = torch.zeros_like(prices, dtype=torch.bool, device=device)
            entry_prices = torch.zeros_like(prices, device=device)
            pnl = torch.zeros_like(prices, device=device)
            equity = torch.ones_like(prices, device=device) * self.initial_capital

            # Simulazione
            for i in range(1, len(prices)):
                current_price = prices[i]
                
                mask_entry = entry_conditions[i] & ~position_active[i-1]
                mask_active = position_active[i-1]
                
                # Entry
                position_active[i:][mask_entry] = True
                entry_prices[i:][mask_entry] = current_price
                
                # Check exit
                if torch.any(mask_active):
                    entry_price = entry_prices[i-1][mask_active]
                    price_change = (current_price - entry_price) / entry_price
                    
                    exit_mask = (
                        (price_change <= -stop_loss_pct) |  # Stop loss
                        (price_change >= take_profit_pct) |  # Take profit
                        entry_conditions[i][mask_active]  # New signal
                    )
                    
                    if torch.any(exit_mask):
                        position_active[i:][mask_active][exit_mask] = False
                        pnl[i][mask_active][exit_mask] = (
                            price_change[exit_mask] * position_size_pct * equity[i-1]
                        )

                # Update equity
                equity[i] = equity[i-1] + pnl[i]

            # Calcola metriche
            total_trades = torch.sum(pnl != 0, dim=0)
            winning_trades = torch.sum(pnl > 0, dim=0)
            
            metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": winning_trades.float() / total_trades.float(),
                "total_pnl": equity[-1] - self.initial_capital,
                "final_capital": equity[-1],
                "max_drawdown": torch.max((torch.maximum.accumulate(equity) - equity) / 
                                         torch.maximum.accumulate(equity)),
                "sharpe_ratio": self._calculate_sharpe_torch(equity),
                "profit_factor": self._calculate_profit_factor_torch(pnl)
            }

            return metrics
            
        except Exception as e:
            logger.error(f"Error in vectorized simulation: {e}")
            return {
                "total_trades": torch.tensor(0),
                "winning_trades": torch.tensor(0),
                "win_rate": torch.tensor(0.0),
                "total_pnl": torch.tensor(0.0),
                "final_capital": torch.tensor(self.initial_capital),
                "max_drawdown": torch.tensor(0.0),
                "sharpe_ratio": torch.tensor(0.0),
                "profit_factor": torch.tensor(0.0)
            }

    def _calculate_sharpe_torch(self: TradingSimulator, 
                              equity: torch.Tensor) -> torch.Tensor:
        """
        Calcola Sharpe ratio usando PyTorch
        
        Args:
            equity: Serie temporale dell'equity
            
        Returns:
            Sharpe ratio
        """
        try:
            returns = (equity[1:] - equity[:-1]) / equity[:-1]
            if len(returns) == 0:
                return torch.tensor(0.0, device=equity.device)
            
            std = torch.std(returns)
            if std == 0:
                return torch.tensor(0.0, device=equity.device)
                
            return torch.sqrt(torch.tensor(252.0)) * (torch.mean(returns) / std)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return torch.tensor(0.0, device=equity.device)

    def _calculate_profit_factor_torch(self: TradingSimulator, 
                                     pnl: torch.Tensor) -> torch.Tensor:
        """
        Calcola profit factor usando PyTorch
        
        Args:
            pnl: Serie temporale del P&L
            
        Returns:
            Profit factor
        """
        try:
            profits = torch.sum(torch.where(pnl > 0, pnl, torch.tensor(0.0, device=pnl.device)))
            losses = torch.abs(torch.sum(torch.where(pnl < 0, pnl, torch.tensor(0.0, device=pnl.device))))
            
            return profits / losses if losses != 0 else torch.tensor(0.0, device=pnl.device)
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return torch.tensor(0.0, device=pnl.device)

    # Aggiungi metodi al TradingSimulator
    simulator_class.run_simulation_vectorized_torch = run_simulation_vectorized_torch
    simulator_class._calculate_sharpe_torch = _calculate_sharpe_torch
    simulator_class._calculate_profit_factor_torch = _calculate_profit_factor_torch
