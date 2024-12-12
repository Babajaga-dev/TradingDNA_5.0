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
        self.use_gpu = config.get("genetic.optimizer.use_gpu", False)
        self.element_size_bytes = config.get("simulator.metrics.element_size_bytes", 32)
        self.setup_device(config)
        self.setup_memory_config(config)
        self.setup_parallel_config(config)
        self.setup_batch_config(config)
        
        logger.info(f"SimulatorDevice initialized with GPU: {self.use_gpu}")
        if self.use_gpu:
            logger.info(f"Using device: {torch.cuda.get_device_name(0)}")
            logger.info(f"Mixed precision: {self.mixed_precision}")
            logger.info(f"Memory reserve: {self.memory_reserve}MB")
            logger.info(f"Element size: {self.element_size_bytes} bytes")

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
                
                # Mixed precision
                self.mixed_precision = config.get("genetic.optimizer.device_config.mixed_precision", False)
                if self.mixed_precision:
                    self.scaler = torch.amp.GradScaler('cuda')
                    
                # Setup CUDA streams per overlap transfers
                if self.parallel_config["async_loading"]:
                    try:
                        num_streams = self.parallel_config["cuda_streams"]
                        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
                        logger.info(f"Created {num_streams} CUDA streams")
                    except Exception as e:
                        logger.error(f"Error creating CUDA streams: {str(e)}")
                        self.streams = []
                    
                # Pin memory se richiesto
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
        """Setup configurazione batch processing"""
        batch_config = config.get("genetic.batch_processing", {})
        
        self.batch_config = {
            "enabled": batch_config.get("enabled", True),
            "adaptive": batch_config.get("adaptive_batching", True),
            "min_size": batch_config.get("min_batch_size", 16384),
            "max_size": batch_config.get("max_batch_size", 65536),
            "memory_limit": batch_config.get("memory_limit", 7900),
            "prefetch": batch_config.get("prefetch_factor", 6),
            "overlap": batch_config.get("overlap_transfers", True)
        }
        
        # Validazione parametri
        self._validate_batch_config()
        self._log_batch_config()

    def _validate_batch_config(self):
        """Valida i parametri di batch processing"""
        # Validazione min/max size
        if self.batch_config["min_size"] > self.batch_config["max_size"]:
            logger.warning("min_batch_size > max_batch_size, correggendo...")
            self.batch_config["min_size"] = min(16384, self.batch_config["max_size"])
            
        # Validazione prefetch_factor
        if self.batch_config["prefetch"] < 1:
            logger.warning("prefetch_factor < 1, impostando a 2")
            self.batch_config["prefetch"] = 2
        elif self.batch_config["prefetch"] > 10:
            logger.warning("prefetch_factor troppo alto, limitando a 10")
            self.batch_config["prefetch"] = 10
            
        # Validazione memory_limit
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
        """
        Determina la dimensione ottimale del batch in base alla configurazione
        e alle risorse disponibili
        
        Args:
            data_size: Dimensione dei dati da processare
            
        Returns:
            Dimensione ottimale del batch
        """
        if not self.batch_config["enabled"] or not self.batch_config["adaptive"]:
            return self.batch_config["max_size"]
            
        try:
            if self.use_gpu:
                # Calcolo basato sulla memoria disponibile
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_limit = self.batch_config["memory_limit"] / 1024  # Converti in GB
                available_memory = max(0, memory_limit - memory_allocated)
                
                # Considera il prefetch factor nel calcolo
                prefetch_overhead = self.batch_config["prefetch"] / 10
                available_memory *= (1 - prefetch_overhead)
                
                # Usa element_size_bytes per calcolo più preciso
                estimated_size = min(
                    self.batch_config["max_size"],
                    max(
                        self.batch_config["min_size"],
                        int(available_memory * 1024**3 / (self.element_size_bytes * data_size))
                    )
                )
                
                # Aggiusta per overlap transfers se abilitato
                if self.batch_config["overlap"]:
                    estimated_size = int(estimated_size * 0.9)  # Riserva 10% per overlap
                    
                return estimated_size
            else:
                return self.batch_config["min_size"]
                
        except Exception as e:
            logger.error(f"Error calculating batch size: {str(e)}")
            return self.batch_config["min_size"]

    def prefetch_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Prefetch dei dati in memoria device se possibile
        
        Args:
            data: Tensor da prefetch
            
        Returns:
            Tensor prefetchato
        """
        try:
            if not self.batch_config["enabled"] or not self.batch_config["overlap"]:
                return data.to(self.device)
                
            if self.use_gpu:
                # Usa stream dedicato per prefetch
                stream = self.get_stream(0)
                if stream is not None:
                    with torch.cuda.stream(stream):
                        prefetched = data.to(self.device, non_blocking=True)
                    # Sincronizza solo se necessario
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
                # Configura cache per batch processing
                if self.batch_config["overlap"]:
                    torch.cuda.set_stream(self.get_stream(0))
                    
                # Riserva memoria per prefetch se necessario
                if self.batch_config["prefetch"] > 1:
                    torch.cuda.empty_cache()
                    
            yield
            
        finally:
            if self.use_gpu:
                # Ripristina stream principale
                torch.cuda.set_stream(torch.cuda.default_stream())
                # Sincronizza se necessario
                if self.batch_config["overlap"]:
                    torch.cuda.synchronize()

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

    def cleanup(self):
        """Pulizia risorse device"""
        if self.use_gpu:
            torch.cuda.empty_cache()
            for stream in self.streams:
                stream.synchronize()
