import torch
import torch.cuda
import numpy as np
import logging
from typing import List, Optional, ContextManager
from contextlib import nullcontext
import gc

logger = logging.getLogger(__name__)

class SimulatorDevice:
    def __init__(self, config):
        self.use_gpu = config.get("genetic.optimizer.use_gpu", False)
        self.setup_device(config)
        self.setup_memory_config(config)
        self.setup_parallel_config(config)
        self.setup_batch_config(config)

    def setup_device(self, config):
        """Setup del dispositivo (GPU/CPU)"""
        if self.use_gpu and torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
                self.device = torch.device("cuda")
                self.num_gpus = torch.cuda.device_count()
                logger.info(f"Using {self.num_gpus} CUDA devices")
                
                # Configura precisione
                self.precision = config.get("genetic.optimizer.device_config.precision", "float32")
                self.dtype = torch.float16 if self.precision == "float16" else torch.float32
                
                # Configura memoria
                self.memory_reserve = config.get("genetic.optimizer.device_config.memory_reserve", 2048)
                self.max_batch_size = config.get("genetic.optimizer.device_config.max_batch_size", 1024)
                
                # Setup CUDA streams
                if self.parallel_config["async_loading"]:
                    try:
                        num_streams = self.parallel_config["cuda_streams"]
                        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
                        logger.info(f"Created {num_streams} CUDA streams")
                    except Exception as e:
                        logger.error(f"Error creating CUDA streams: {str(e)}")
                        self.streams = []
                
                # Mixed precision
                self.mixed_precision = config.get("genetic.optimizer.device_config.mixed_precision", False)
                if self.mixed_precision:
                    self.scaler = torch.amp.GradScaler('cuda')
                    
                # Pin memory
                if self.parallel_config["pin_memory"]:
                    torch.cuda.set_rng_state(torch.cuda.get_rng_state())
                    
            except Exception as e:
                logger.error(f"Error setting up CUDA: {str(e)}")
                self._setup_cpu_fallback(config)
        else:
            self._setup_cpu_fallback(config)

    def _setup_cpu_fallback(self, config):
        """Setup CPU come fallback"""
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.max_batch_size = config.get("genetic.batch_size", 32)
        logger.info("Using CPU device")
        self.mixed_precision = False
        self.streams = []

    def setup_memory_config(self, config):
        """Setup configurazione memoria"""
        self.memory_config = {
            "prealloc": config.get("genetic.memory_management.preallocation", True),
            "cache_mode": config.get("genetic.memory_management.cache_mode", "all"),
            "release_thresh": config.get("genetic.memory_management.release_threshold", 0.99),
            "defrag_thresh": config.get("genetic.memory_management.defrag_threshold", 0.9),
            "periodic_gc": config.get("genetic.memory_management.periodic_gc", False)
        }

    def setup_parallel_config(self, config):
        """Setup configurazione parallela"""
        self.parallel_config = {
            "chunk_size": config.get("genetic.parallel_config.chunk_size", 512),
            "cuda_streams": config.get("genetic.parallel_config.cuda_streams", 4),
            "async_loading": config.get("genetic.parallel_config.async_loading", True),
            "pin_memory": config.get("genetic.parallel_config.pin_memory", True),
            "persistent_workers": config.get("genetic.parallel_config.persistent_workers", True)
        }

    def setup_batch_config(self, config):
        """Setup configurazione batch"""
        self.batch_config = {
            "enabled": config.get("genetic.batch_processing.enabled", True),
            "adaptive": config.get("genetic.batch_processing.adaptive_batching", True),
            "min_size": config.get("genetic.batch_processing.min_batch_size", 16384),
            "max_size": config.get("genetic.batch_processing.max_batch_size", 65536),
            "memory_limit": config.get("genetic.batch_processing.memory_limit", 7900),
            "prefetch": config.get("genetic.batch_processing.prefetch_factor", 6),
            "overlap": config.get("genetic.batch_processing.overlap_transfers", True)
        }

    def get_stream(self, index: int) -> ContextManager:
        """Ottiene uno stream CUDA o un contesto nullo"""
        if self.streams and index < len(self.streams):
            return torch.cuda.stream(self.streams[index])
        return nullcontext()

    def manage_memory(self) -> None:
        """Gestisce la memoria secondo la configurazione"""
        if not self.use_gpu:
            return
            
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            
            if memory_allocated > self.memory_config["release_thresh"] * memory_reserved:
                torch.cuda.empty_cache()
                
            if memory_allocated > self.memory_config["defrag_thresh"] * memory_reserved:
                if self.memory_config["periodic_gc"]:
                    gc.collect()
                    
            if self.memory_config["prealloc"]:
                if memory_reserved < self.batch_config["memory_limit"]:
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Error in memory management: {str(e)}")

    def get_optimal_batch_size(self, data_size: int) -> int:
        """Determina batch size ottimale"""
        if not self.batch_config["adaptive"]:
            return self.batch_config["max_size"]
            
        try:
            if self.use_gpu:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_limit = self.batch_config["memory_limit"]
                available_memory = max(0, memory_limit - memory_allocated)
                
                estimated_size = min(
                    self.batch_config["max_size"],
                    max(
                        self.batch_config["min_size"],
                        int(available_memory * 1024**3 / (32 * data_size))
                    )
                )
                
                return estimated_size
            else:
                return self.batch_config["min_size"]
                
        except Exception as e:
            logger.error(f"Error calculating batch size: {str(e)}")
            return self.batch_config["min_size"]

    def to_tensor(self, data: any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Converte i dati in tensor con gestione errori"""
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
            logger.error(f"Error converting to tensor: {str(e)}")
            raise
