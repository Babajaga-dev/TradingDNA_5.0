import logging
import os
import psutil
import torch
import traceback
import gc
import time
import numpy as np
from typing import List, Tuple, Optional
from contextlib import contextmanager, nullcontext

logger = logging.getLogger(__name__)

class DeviceManager:
    def __init__(self, config):
        self.config = config  # Aggiungo l'assegnazione di config
        self.use_gpu = config.get("genetic.optimizer.use_gpu", False)
        self.gpu_backend = config.get("genetic.optimizer.gpu_backend", "auto")
        self.precision = config.get("genetic.optimizer.device_config.precision", "float32")
        self.dtype = torch.float16 if self.precision == "float16" else torch.float32
        
        # Parametri memory management
        memory_config = config.get("genetic.memory_management", {})
        self.cache_mode = memory_config.get("cache_mode", "auto")
        self.defrag_threshold = memory_config.get("defrag_threshold", 0.7)
        self.periodic_gc = memory_config.get("periodic_gc", True)
        self.gc_interval = memory_config.get("gc_interval", 300)
        self.last_gc_time = time.time()
        
        # Parametri batch processing
        batch_config = config.get("genetic.batch_processing", {})
        self.batch_config = {
            "enabled": batch_config.get("enabled", True),
            "adaptive": batch_config.get("adaptive_batching", True),
            "min_size": batch_config.get("min_batch_size", 16384),
            "max_size": batch_config.get("max_batch_size", 65536),
            "memory_limit": batch_config.get("memory_limit", 3072),
            "prefetch": batch_config.get("prefetch_factor", 2),
            "overlap": batch_config.get("overlap_transfers", True)
        }
        
        try:
            if not self.use_gpu:
                self._setup_cpu(config)
            elif self.gpu_backend == "arc":
                # Import Intel extension solo se necessario
                try:
                    import intel_extension_for_pytorch as ipex
                    if torch.xpu.is_available():
                        self._setup_xpu(config)
                    else:
                        logger.warning("Intel XPU not available")
                        self._setup_cpu(config)
                except ImportError:
                    logger.warning("Intel Extension for PyTorch not found")
                    self._setup_cpu(config)
            elif (self.gpu_backend == "cuda" or self.gpu_backend == "auto") and torch.cuda.is_available():
                self._setup_gpu(config)
            else:
                logger.warning(f"Requested backend {self.gpu_backend} not available")
                self._setup_cpu(config)
                
        except Exception as e:
            logger.error(f"Error during device setup: {str(e)}")
            logger.error(traceback.format_exc())
            self._setup_cpu(config)

    def _validate_compute_capability(self, device_idx: int, required_cc: str) -> bool:
        """Valida la compute capability del dispositivo"""
        try:
            req_major, req_minor = map(int, required_cc.split("."))
            props = torch.cuda.get_device_properties(device_idx)
            dev_major, dev_minor = props.major, props.minor
            
            if dev_major > req_major:
                return True
            elif dev_major == req_major:
                return dev_minor >= req_minor
            return False
        except Exception as e:
            logger.warning(f"Error validating compute capability: {e}")
            return True

    def _apply_optimization_level(self, level: int, backend: str) -> None:
        """Applica il livello di ottimizzazione specificato"""
        try:
            if not 0 <= level <= 3:
                logger.warning(f"Invalid optimization level: {level}. Using default (3)")
                level = 3
                
            if backend == "cuda":
                if level >= 1:
                    torch.backends.cudnn.enabled = True
                    logger.info("CuDNN enabled")
                    
                if level >= 2:
                    torch.backends.cudnn.benchmark = True
                    logger.info("CuDNN benchmark mode enabled")
                    
                if level == 3:
                    if hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = True
                    if hasattr(torch.backends.cuda, 'matmul'):
                        torch.backends.cuda.matmul.allow_tf32 = True
                    if hasattr(torch, 'set_float32_matmul_precision'):
                        torch.set_float32_matmul_precision('high')
                    logger.info("Level 3 optimizations enabled (including TF32 if supported)")
            
            elif backend == "xpu":
                if level >= 1:
                    # Ottimizzazioni base per XPU
                    logger.info("XPU base optimizations enabled")
                    
                if level >= 2:
                    # Abilita ottimizzazioni aggiuntive per XPU
                    if hasattr(torch.xpu, 'enable_optimizations'):
                        torch.xpu.enable_optimizations()
                        logger.info("XPU additional optimizations enabled")
                    
        except Exception as e:
            logger.error(f"Error applying optimization level: {e}")

    def _setup_xpu(self, config):
        """Setup per Intel XPU"""
        self.devices = [torch.device("xpu")]
        self.num_gpus = 1
        
        # Setup memoria GPU
        arc_config = self.config.get("genetic.optimizer.device_config.arc", {})
        self.memory_reserve = arc_config.get("memory_reserve", 1024)
        self.max_batch_size = arc_config.get("max_batch_size", 65536)
        
        # Mixed precision
        self.mixed_precision = arc_config.get("mixed_precision", True)
        if self.mixed_precision:
            # Configura il modello per FP16
            self.dtype = torch.float16
            logger.info("XPU mixed precision enabled")
            
        # Configurazione stream
        num_streams = min(2, self.config.get("genetic.parallel_config.xpu_streams", 2))
        self.streams = [torch.xpu.Stream() for _ in range(num_streams)]
        
        # Memory strategy
        self.memory_strategy = arc_config.get("memory_strategy", {})
        
        # Applica ottimizzazioni XPU
        opt_level = arc_config.get("optimization_level", 3)
        self._apply_optimization_level(opt_level, "xpu")
        
        logger.info("Intel Arc GPU configuration completed")

    def _setup_gpu(self, config):
        """Setup per NVIDIA GPU"""
        # Verifica compute capability
        required_cc = config.get("genetic.optimizer.cuda_config.compute_capability", "6.1")
        available_gpus = []
        
        for i in range(torch.cuda.device_count()):
            if self._validate_compute_capability(i, required_cc):
                available_gpus.append(i)
                
        if not available_gpus:
            logger.warning(f"No GPU meets required compute capability {required_cc}")
            self._setup_cpu(config)
            return
            
        self.num_gpus = len(available_gpus)
        self.devices = [torch.device(f"cuda:{i}") for i in available_gpus]
        logger.info(f"Using {self.num_gpus} CUDA devices")
        
        # Setup memoria GPU
        cuda_config = self.config.get("genetic.optimizer.device_config.cuda", {})
        self.memory_reserve = cuda_config.get("memory_reserve", 2048)
        self.max_batch_size = cuda_config.get("max_batch_size", 1024)
        
        # Mixed precision
        self.mixed_precision = cuda_config.get("mixed_precision", True)
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler()
            
        self._setup_cuda_config(config)
        self._setup_linux_specific(config)
        self._log_device_info()

    def _setup_cpu(self, config):
        """Setup per CPU"""
        self.devices = [torch.device("cpu")]
        self.num_gpus = 1
        self.mixed_precision = False
        
        cpu_threads = config.get("genetic.optimizer.torch_threads", 4)
        torch.set_num_threads(cpu_threads)
        logger.info(f"Using CPU with {cpu_threads} threads")

    def _setup_cuda_config(self, config):
        """Setup configurazione CUDA"""
        cuda_config = config.get("genetic.optimizer.cuda_config", {})
        
        # Applica livello ottimizzazione
        opt_level = cuda_config.get("optimization_level", 3)
        self._apply_optimization_level(opt_level, "cuda")
        
        # Configura TF32
        allow_tf32 = cuda_config.get("allow_tf32", True)
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        
        # Configura benchmark e deterministic
        benchmark = cuda_config.get("benchmark", True)
        deterministic = cuda_config.get("deterministic", False)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
        
        # Strategia memoria
        memory_config = config.get("genetic.memory_management", {})
        self.cuda_config = {
            "allow_tf32": allow_tf32,
            "benchmark": benchmark,
            "deterministic": deterministic,
            "memory_strategy": {
                "preallocate": memory_config.get("preallocation", True),
                "empty_cache_threshold": memory_config.get("release_threshold", 0.99),
                "force_release_threshold": memory_config.get("release_threshold", 0.99)
            },
            "compute_capability": cuda_config.get("compute_capability", "6.1"),
            "optimization_level": opt_level
        }

    def _setup_linux_specific(self, config):
        """Setup specifico per Linux"""
        if os.name == 'posix':
            linux_config = config.get("genetic.optimizer.linux_specific", {})
            if linux_config.get("process_affinity", False):
                try:
                    process = psutil.Process()
                    process.cpu_affinity([0])
                    logger.info("Process affinity set")
                except Exception as e:
                    logger.warning(f"Could not set process affinity: {str(e)}")
                    
            if linux_config.get("shared_memory", False):
                try:
                    torch.multiprocessing.set_sharing_strategy('file_system')
                    logger.info("Shared memory strategy set to file_system")
                except Exception as e:
                    logger.warning(f"Could not set shared memory strategy: {str(e)}")

    def _log_device_info(self):
        """Log informazioni sui dispositivi"""
        logger.info("\nDevice Configuration:")
        logger.info(f"Backend: {self.gpu_backend}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"Cache mode: {self.cache_mode}")
        logger.info(f"Defrag threshold: {self.defrag_threshold}")
        logger.info(f"Periodic GC: {self.periodic_gc}")
        logger.info(f"GC interval: {self.gc_interval} seconds")
        
        for i, device in enumerate(self.devices):
            if device.type == "cuda":
                try:
                    device_name = torch.cuda.get_device_name(device.index)
                    props = torch.cuda.get_device_properties(device.index)
                    memory_allocated = torch.cuda.memory_allocated(device.index) / 1e9
                    memory_total = props.total_memory / 1e9
                    
                    logger.info(f"\nGPU {device.index} ({device_name}):")
                    logger.info(f"- Compute capability: {props.major}.{props.minor}")
                    logger.info(f"- Total memory: {memory_total:.2f} GB")
                    logger.info(f"- Memory allocated: {memory_allocated:.2f} GB")
                    logger.info(f"- Memory reserved: {self.memory_reserve / 1024:.1f} MB")
                except Exception as e:
                    logger.warning(f"Could not get info for GPU {device.index}: {str(e)}")
            elif device.type == "xpu":
                try:
                    memory_allocated = torch.xpu.memory_allocated() / 1e9
                    memory_total = torch.xpu.get_device_properties().total_memory / 1e9
                    
                    logger.info("\nIntel Arc GPU:")
                    logger.info(f"- Total memory: {memory_total:.2f} GB")
                    logger.info(f"- Memory allocated: {memory_allocated:.2f} GB")
                    logger.info(f"- Memory reserved: {self.memory_reserve / 1024:.1f} MB")
                except Exception as e:
                    logger.warning(f"Could not get info for XPU: {str(e)}")

    def manage_memory(self) -> None:
        """Gestisce la memoria secondo la configurazione"""
        if not self.use_gpu:
            return
            
        try:
            if self.gpu_backend == "arc":
                memory_allocated = torch.xpu.memory_allocated() / 1024**3
                memory_reserved = torch.xpu.memory_reserved() / 1024**3
            else:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
            
            # Gestione memoria basata sulla strategia del backend
            if self.memory_strategy.get("preallocate", False):
                prealloc_thresh = self.memory_strategy.get("prealloc_threshold", 0.4)
                if memory_allocated < prealloc_thresh * memory_reserved:
                    if self.gpu_backend == "arc":
                        torch.xpu.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                    
            # Gestione cache
            empty_cache_thresh = self.memory_strategy.get("empty_cache_threshold", 0.8)
            if memory_allocated > empty_cache_thresh * memory_reserved:
                if self.gpu_backend == "arc":
                    torch.xpu.empty_cache()
                else:
                    torch.cuda.empty_cache()
                
            # Garbage collection
            if self.periodic_gc:
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
                if self.gpu_backend == "arc":
                    memory_allocated = torch.xpu.memory_allocated() / 1024**3
                else:
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
                        int(available_memory * 1024**3 / (32 * data_size))  # Assumiamo 32 bytes per elemento
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
                return data.to(self.devices[0])  # Usa il primo device disponibile
                
            if self.use_gpu:
                stream = self.streams[0] if hasattr(self, 'streams') else None
                if stream is not None:
                    with stream:
                        prefetched = data.to(self.devices[0], non_blocking=True)
                    if self.batch_config["overlap"]:
                        stream.synchronize()
                    return prefetched
                    
            return data.to(self.devices[0])
            
        except Exception as e:
            logger.error(f"Error in data prefetch: {str(e)}")
            return data.to(self.devices[0])

    @contextmanager
    def batch_processing_context(self):
        """Context manager per il batch processing"""
        try:
            if self.use_gpu and self.batch_config["enabled"]:
                if self.batch_config["overlap"] and hasattr(self, 'streams'):
                    if self.gpu_backend == "arc":
                        torch.xpu.set_stream(self.streams[0])
                    else:
                        torch.cuda.set_stream(self.streams[0])
                    
                if self.batch_config["prefetch"] > 1:
                    if self.gpu_backend == "arc":
                        torch.xpu.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                    
            yield
            
        finally:
            if self.use_gpu:
                if self.gpu_backend == "arc":
                    torch.xpu.set_stream(torch.xpu.default_stream())
                    if self.batch_config["overlap"]:
                        torch.xpu.synchronize()
                else:
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
                
            return tensor.to(device=self.devices[0])
            
        except Exception as e:
            logger.error(f"Error converting to tensor: {str(e)}")
            raise

    def cleanup(self):
        """Pulizia risorse device"""
        if self.use_gpu:
            try:
                # Sincronizza e libera stream
                if hasattr(self, 'streams'):
                    for stream in self.streams:
                        stream.synchronize()
                
                # Libera memoria
                if self.gpu_backend == "arc":
                    torch.xpu.empty_cache()
                else:
                    torch.cuda.empty_cache()
                
                # Reset device
                if hasattr(self, 'scaler'):
                    del self.scaler
                    
            except Exception as e:
                logger.error(f"Error in cleanup: {str(e)}")
