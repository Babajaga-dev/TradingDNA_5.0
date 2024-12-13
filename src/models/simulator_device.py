# src/models/simulator_device.py
import torch
import gc
import logging
import numpy as np
from typing import Dict, Optional, List, ContextManager
from contextlib import contextmanager, nullcontext

logger = logging.getLogger(__name__)

class SimulatorDevice:
    def __init__(self, config):
        self.config = config
        self.gpu_backend = config.get("genetic.optimizer.gpu_backend", "auto")
        self.use_gpu = config.get("genetic.optimizer.use_gpu", False)
        self.element_size_bytes = config.get("simulator.metrics.element_size_bytes", 32)
        self.setup_device()
        self.setup_memory_config()
        self.setup_parallel_config()
        self.setup_batch_config()
        
        logger.info(f"SimulatorDevice initialized with backend: {self.gpu_backend}")
        if self.use_gpu:
            logger.info(f"Using device: {torch.cuda.get_device_name(0)}")
            logger.info(f"Mixed precision: {self.mixed_precision}")
            logger.info(f"Memory reserve: {self.memory_reserve}MB")
            logger.info(f"Element size: {self.element_size_bytes} bytes")

    def detect_gpu_backends(self) -> Dict[str, bool]:
        """Rileva i backend GPU disponibili"""
        backends = {
            'cuda': False,
            'arc': False
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                if 'NVIDIA' in props.name:
                    backends['cuda'] = True
                if 'Intel' in props.name and 'Arc' in props.name:
                    backends['arc'] = True
                    
        return backends

    def setup_device(self):
        """Setup del dispositivo (GPU/CPU)"""
        if not self.use_gpu:
            self._setup_cpu_fallback()
            return

        backends = self.detect_gpu_backends()
        
        # Selezione automatica backend
        if self.gpu_backend == "auto":
            if backends['arc']:
                self.gpu_backend = "arc"
            elif backends['cuda']:
                self.gpu_backend = "cuda"
            else:
                self.gpu_backend = "cpu"
                self._setup_cpu_fallback()
                return

        # Setup backend specifico
        try:
            if self.gpu_backend == "arc" and backends['arc']:
                self._setup_arc_config()
            elif self.gpu_backend == "cuda" and backends['cuda']:
                self._setup_cuda_config()
            else:
                logger.warning(f"Requested backend {self.gpu_backend} not available")
                self._setup_cpu_fallback()
                
        except Exception as e:
            logger.error(f"Error setting up {self.gpu_backend}: {str(e)}")
            self._setup_cpu_fallback()

    def _setup_arc_config(self):
        """Setup configurazione Intel Arc"""
        torch.cuda.set_device(0)
        self.device = torch.device("cuda")
        
        arc_config = self.config.get("genetic.optimizer.device_config.arc", {})
        
        # Configurazioni ottimizzate per Arc
        self.dtype = torch.float16  # Arc preferisce FP16
        self.memory_reserve = arc_config.get("memory_reserve", 1024)
        self.max_batch_size = arc_config.get("max_batch_size", 65536)
        self.mixed_precision = arc_config.get("mixed_precision", True)
        
        # Mixed precision setup
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler()
            
        # Configurazione stream
        num_streams = min(2, self.config.get("genetic.parallel_config.cuda_streams", 2))
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        
        # Memory strategy
        self.memory_strategy = arc_config.get("memory_strategy", {})
        
        logger.info("Intel Arc GPU configuration completed")

    def _setup_cuda_config(self):
        """Setup configurazione NVIDIA CUDA"""
        torch.cuda.set_device(0)
        self.device = torch.device("cuda")
        
        cuda_config = self.config.get("genetic.optimizer.device_config.cuda", {})
        
        # Configurazioni ottimizzate per CUDA
        self.dtype = torch.float32
        self.memory_reserve = cuda_config.get("memory_reserve", 2048)
        self.max_batch_size = cuda_config.get("max_batch_size", 131072)
        self.mixed_precision = cuda_config.get("mixed_precision", True)
        
        # Mixed precision setup
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler()
            
        # Configurazione CUDA
        cuda_settings = self.config.get("genetic.optimizer.cuda_config", {})
        torch.backends.cuda.matmul.allow_tf32 = cuda_settings.get("allow_tf32", True)
        torch.backends.cudnn.benchmark = cuda_settings.get("benchmark", True)
        torch.backends.cudnn.deterministic = cuda_settings.get("deterministic", False)
        
        # Stream configuration
        num_streams = self.config.get("genetic.parallel_config.cuda_streams", 4)
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        
        # Memory strategy
        self.memory_strategy = cuda_config.get("memory_strategy", {})
        
        logger.info("NVIDIA CUDA configuration completed")

    def _setup_cpu_fallback(self):
        """Setup CPU come fallback"""
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.max_batch_size = self.config.get("genetic.batch_size", 32)
        self.mixed_precision = False
        self.streams = []
        self.memory_strategy = {}
        logger.info("Using CPU device (fallback)")

    def setup_memory_config(self):
        """Setup configurazione memoria"""
        self.memory_config = {
            "prealloc": self.config.get("genetic.memory_management.preallocation", True),
            "cache_mode": self.config.get("genetic.memory_management.cache_mode", "balanced"),
            "release_thresh": self.config.get("genetic.memory_management.release_threshold", 0.8),
            "defrag_thresh": self.config.get("genetic.memory_management.defrag_threshold", 0.6),
            "periodic_gc": self.config.get("genetic.memory_management.periodic_gc", True),
            "gc_interval": self.config.get("genetic.memory_management.gc_interval", 90)
        }

    def setup_parallel_config(self):
        """Setup configurazione parallela"""
        self.parallel_config = {
            "chunk_size": self.config.get("genetic.parallel_config.chunk_size", 128),
            "async_loading": self.config.get("genetic.parallel_config.async_loading", True),
            "pin_memory": self.config.get("genetic.parallel_config.pin_memory", True),
            "persistent_workers": self.config.get("genetic.parallel_config.persistent_workers", True)
        }

    def setup_batch_config(self):
        """Setup configurazione batch processing"""
        batch_config = self.config.get("genetic.batch_processing", {})
        
        self.batch_config = {
            "enabled": batch_config.get("enabled", True),
            "adaptive": batch_config.get("adaptive_batching", True),
            "min_size": batch_config.get("min_batch_size", 16384),
            "max_size": batch_config.get("max_batch_size", 65536),
            "memory_limit": batch_config.get("memory_limit", 3072),
            "prefetch": batch_config.get("prefetch_factor", 2),
            "overlap": batch_config.get("overlap_transfers", True)
        }
        
        self._validate_batch_config()
        self._log_batch_config()

    def _validate_batch_config(self):
        """Valida i parametri di batch processing"""
        if self.batch_config["min_size"] > self.batch_config["max_size"]:
            logger.warning("min_batch_size > max_batch_size, correggendo...")
            self.batch_config["min_size"] = min(16384, self.batch_config["max_size"])
            
        if self.batch_config["prefetch"] < 1:
            logger.warning("prefetch_factor < 1, impostando a 2")
            self.batch_config["prefetch"] = 2
        elif self.batch_config["prefetch"] > 10:
            logger.warning("prefetch_factor troppo alto, limitando a 10")
            self.batch_config["prefetch"] = 10
            
        if self.batch_config["memory_limit"] < 1000:
            logger.warning("memory_limit troppo basso, impostando a 1000MB")
            self.batch_config["memory_limit"] = 1000

    def _log_batch_config(self):
        """Log della configurazione batch"""
        logger.info("Batch Processing Configuration:")
        logger.info(f"Enabled: {self.batch_config['enabled']}")
        logger.info(f"Adaptive batching: {self.batch_config['adaptive']}")
        logger.info(f"Min batch size: {self.batch_config['min_size']}")
        logger.info(f"Max batch size: {self.batch_config['max_size']}")
        logger.info(f"Memory limit: {self.batch_config['memory_limit']}MB")
        logger.info(f"Prefetch factor: {self.batch_config['prefetch']}")
        logger.info(f"Overlap transfers: {self.batch_config['overlap']}")

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
            
            # Gestione memoria basata sulla strategia del backend
            if self.memory_strategy.get("preallocate", False):
                prealloc_thresh = self.memory_strategy.get("prealloc_threshold", 0.4)
                if memory_allocated < prealloc_thresh * memory_reserved:
                    torch.cuda.empty_cache()
                    
            # Gestione cache
            empty_cache_thresh = self.memory_strategy.get("empty_cache_threshold", 0.8)
            if memory_allocated > empty_cache_thresh * memory_reserved:
                torch.cuda.empty_cache()
                
            # Garbage collection
            if self.memory_config["periodic_gc"]:
                force_release_thresh = self.memory_strategy.get("force_release_threshold", 0.9)
                if memory_allocated > force_release_thresh * memory_reserved:
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"Error in memory management: {str(e)}")

    def get_optimal_batch_size(self, data_size: int) -> int:
        """Determina la dimensione ottimale del batch"""
        if not self.batch_config["enabled"] or not self.batch_config["adaptive"]:
            return self.batch_config["max_size"]
            
        try:
            if self.use_gpu:
                # Calcolo basato sulla memoria disponibile
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_limit = self.batch_config["memory_limit"] / 1024
                available_memory = max(0, memory_limit - memory_allocated)
                
                # Considera il prefetch factor
                prefetch_overhead = self.batch_config["prefetch"] / 10
                available_memory *= (1 - prefetch_overhead)
                
                # Calcolo dimensione batch
                estimated_size = min(
                    self.batch_config["max_size"],
                    max(
                        self.batch_config["min_size"],
                        int(available_memory * 1024**3 / (self.element_size_bytes * data_size))
                    )
                )
                
                # Aggiustamento per overlap
                if self.batch_config["overlap"]:
                    estimated_size = int(estimated_size * 0.9)
                    
                return estimated_size
            else:
                return self.batch_config["min_size"]
                
        except Exception as e:
            logger.error(f"Error calculating batch size: {str(e)}")
            return self.batch_config["min_size"]

    def prefetch_data(self, data: torch.Tensor) -> torch.Tensor:
        """Prefetch dei dati in memoria device"""
        try:
            if not self.batch_config["enabled"] or not self.batch_config["overlap"]:
                return data.to(self.device)
                
            if self.use_gpu:
                stream = self.get_stream(0)
                if isinstance(stream, torch.cuda.Stream):
                    with stream:
                        prefetched = data.to(self.device, non_blocking=True)
                    if self.batch_config["overlap"]:
                        stream.synchronize()
                    return prefetched
                    
            return data.to(self.device)
            
        except Exception as e:
            logger.error(f"Error in data prefetch: {str(e)}")
            return data.to(self.device)

    @contextmanager
    def batch_processing_context(self):
        """Context manager per il batch processing"""
        try:
            if self.use_gpu and self.batch_config["enabled"]:
                if self.batch_config["overlap"]:
                    torch.cuda.set_stream(self.get_stream(0))
                    
                if self.batch_config["prefetch"] > 1:
                    torch.cuda.empty_cache()
                    
            yield
            
        finally:
            if self.use_gpu:
                torch.cuda.set_stream(torch.cuda.default_stream())
                if self.batch_config["overlap"]:
                    torch.cuda.synchronize()

    def to_tensor(self, data: any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Converte i dati in tensor"""
        try:
            if isinstance(data, torch.Tensor):
                tensor = data
            elif isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data)
            else:
                tensor = torch.tensor(data)
                
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            else:
                tensor = tensor.to(dtype=self.dtype)
                
            return tensor.to(device=self.device)
            
        except Exception as e:
            logger.error(f"Error converting to tensor: {str(e)}")
            raise

    def cleanup(self):
        """Pulizia risorse device"""
        if self.use_gpu:
            try:
                # Sincronizza e libera stream
                for stream in self.streams:
                    stream.synchronize()
                
                # Libera memoria
                torch.cuda.empty_cache()
                
                # Reset device
                if hasattr(self, 'scaler'):
                    del self.scaler
                    
            except Exception as e:
                logger.error(f"Error in cleanup: {str(e)}")
