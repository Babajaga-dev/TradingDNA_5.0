import torch
import intel_extension_for_pytorch as ipex
import numpy as np
import logging
from typing import Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class TorchDataManager:
    def __init__(self, device: torch.device, config):
        self.device = device
        self.precision = config.get("genetic.optimizer.precision", "float32")
        self.batch_norm = config.get("genetic.optimizer.batch_norm", True)
        self.epsilon = config.get("genetic.optimizer.normalization.epsilon", 1e-8)

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
                    tensor = self._convert_to_tensor(value, dtype)
                    if self.batch_norm:
                        tensor = self._normalize_tensor(tensor)
                    torch_data[key] = tensor

            return torch_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return {}

    def _convert_to_tensor(self, data: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        """
        Converte array numpy in tensor PyTorch
        
        Args:
            data: Array numpy
            dtype: Tipo di dato PyTorch
            
        Returns:
            Tensor PyTorch
        """
        try:
            return torch.tensor(data, dtype=dtype, device=self.device)
        except Exception as e:
            logger.error(f"Error converting to tensor: {e}")
            raise

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalizza un tensor
        
        Args:
            tensor: Tensor da normalizzare
            
        Returns:
            Tensor normalizzato
        """
        try:
            return (tensor - tensor.mean()) / (tensor.std() + self.epsilon)
        except Exception as e:
            logger.error(f"Error normalizing tensor: {e}")
            return tensor

    @contextmanager
    def gpu_memory_manager(self):
        """Context manager per gestire la memoria GPU"""
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "xpu":
                torch.xpu.empty_cache()
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "xpu":
                torch.xpu.empty_cache()

    def to_device(self, data: any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Sposta dati sul device corretto
        
        Args:
            data: Dati da spostare
            dtype: Tipo di dato opzionale
            
        Returns:
            Tensor sul device corretto
        """
        try:
            if isinstance(data, torch.Tensor):
                tensor = data
            elif isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data)
            else:
                tensor = torch.tensor(data)
                
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
                
            return tensor.to(device=self.device)
            
        except Exception as e:
            logger.error(f"Error moving data to device: {e}")
            raise

    def clear_cache(self) -> None:
        """Pulisce la cache della memoria"""
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "xpu":
                torch.xpu.empty_cache()
        except Exception as e:
            logger.error(f"Error clearing device cache: {e}")

    def get_batch_size(self, data_size: int, config) -> int:
        """
        Calcola batch size ottimale
        
        Args:
            data_size: Dimensione dei dati
            config: Configurazione
            
        Returns:
            Batch size ottimale
        """
        try:
            if self.device.type in ["cuda", "xpu"]:
                # Usa batch size basato su memoria GPU disponibile
                memory_limit = config.get("genetic.batch_processing.memory_limit", 7900)
                min_size = config.get("genetic.batch_processing.min_batch_size", 16384)
                max_size = config.get("genetic.batch_processing.max_batch_size", 65536)
                
                if self.device.type == "cuda":
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                else:  # xpu
                    memory_allocated = torch.xpu.memory_allocated() / 1024**3
                    
                available_memory = max(0, memory_limit - memory_allocated)
                
                estimated_size = min(
                    max_size,
                    max(
                        min_size,
                        int(available_memory * 1024**3 / (32 * data_size))
                    )
                )
                
                return estimated_size
            else:
                # Per CPU usa batch size dal config
                return config.get("genetic.batch_size", 32)
                
        except Exception as e:
            logger.error(f"Error calculating batch size: {e}")
            return config.get("genetic.batch_size", 32)
