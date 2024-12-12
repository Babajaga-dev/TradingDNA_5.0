import logging
import os
import psutil
import torch
import traceback
import gc
import time
from typing import List, Tuple

logger = logging.getLogger(__name__)

class DeviceManager:
    def __init__(self, config):
        self.use_gpu = config.get("genetic.optimizer.use_gpu", False)
        self.precision = config.get("genetic.optimizer.device_config.precision", "float32")
        self.dtype = torch.float16 if self.precision == "float16" else torch.float32
        
        # Parametri memory management
        memory_config = config.get("genetic.memory_management", {})
        self.cache_mode = memory_config.get("cache_mode", "auto")
        self.defrag_threshold = memory_config.get("defrag_threshold", 0.7)
        self.periodic_gc = memory_config.get("periodic_gc", True)
        self.gc_interval = memory_config.get("gc_interval", 300)
        self.last_gc_time = time.time()
        
        try:
            if self.use_gpu and torch.cuda.is_available():
                self._setup_gpu(config)
            else:
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

    def _apply_optimization_level(self, level: int) -> None:
        """Applica il livello di ottimizzazione specificato"""
        try:
            if not 0 <= level <= 3:
                logger.warning(f"Invalid optimization level: {level}. Using default (3)")
                level = 3
                
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
                
        except Exception as e:
            logger.error(f"Error applying optimization level: {e}")

    def _setup_gpu(self, config):
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
        self.memory_reserve = config.get("genetic.optimizer.device_config.memory_reserve", 2048)
        self.max_batch_size = config.get("genetic.optimizer.device_config.max_batch_size", 1024)
        
        # Mixed precision
        self.mixed_precision = config.get("genetic.optimizer.device_config.mixed_precision", False)
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler()
            logger.info("Mixed precision training enabled")
            
        self._setup_cuda_config(config)
        self._setup_linux_specific(config)
        self._log_gpu_info()

    def _setup_cpu(self, config):
        self.devices = [torch.device("cpu")]
        self.num_gpus = 1
        self.mixed_precision = False
        
        cpu_threads = config.get("genetic.optimizer.torch_threads", 4)
        torch.set_num_threads(cpu_threads)
        logger.info(f"Using CPU with {cpu_threads} threads")

    def _setup_cuda_config(self, config):
        cuda_config = config.get("genetic.optimizer.cuda_config", {})
        
        # Applica livello ottimizzazione
        opt_level = cuda_config.get("optimization_level", 3)
        self._apply_optimization_level(opt_level)
        
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

    def _log_gpu_info(self):
        logger.info("\nGPU Configuration:")
        logger.info(f"CUDA enabled: {torch.cuda.is_available()}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"CUDA compute capability required: {self.cuda_config['compute_capability']}")
        logger.info(f"CUDA optimization level: {self.cuda_config['optimization_level']}")
        logger.info(f"TF32 enabled: {self.cuda_config['allow_tf32']}")
        logger.info(f"Benchmark mode: {self.cuda_config['benchmark']}")
        logger.info(f"Deterministic mode: {self.cuda_config['deterministic']}")
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

    def _check_memory_fragmentation(self) -> float:
        """Calcola il livello di frammentazione della memoria"""
        if not self.use_gpu:
            return 0.0
            
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            if reserved == 0:
                return 0.0
            return 1.0 - (allocated / reserved)
        except Exception as e:
            logger.error(f"Error checking memory fragmentation: {str(e)}")
            return 0.0

    def _perform_defragmentation(self) -> None:
        """Esegue la deframmentazione della memoria"""
        if not self.use_gpu:
            return
            
        try:
            # Forza il garbage collector
            gc.collect()
            torch.cuda.empty_cache()
            
            # Riallocazione tensori se necessario
            if hasattr(torch.cuda, 'memory_stats'):
                active_blocks = torch.cuda.memory_stats()['active_blocks.all.current']
                if active_blocks > 100:  # soglia arbitraria
                    torch.cuda.empty_cache()
                    logger.info("Memory defragmentation performed")
        except Exception as e:
            logger.error(f"Error during memory defragmentation: {str(e)}")

    def manage_cuda_memory(self) -> None:
        """Gestisce la memoria CUDA in base alla strategia configurata"""
        if not self.use_gpu:
            return
            
        try:
            memory_allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            # Gestione cache in base alla modalità
            if self.cache_mode == "aggressive":
                # Svuota cache più frequentemente
                if memory_allocated > 0.5:
                    torch.cuda.empty_cache()
                    logger.debug("Memory cache emptied (aggressive mode)")
            elif self.cache_mode == "conservative":
                # Mantiene più cache
                if memory_allocated > 0.9:
                    torch.cuda.empty_cache()
                    logger.debug("Memory cache emptied (conservative mode)")
            elif self.cache_mode == "auto":
                # Comportamento adattivo basato sull'uso
                if self.cuda_config["memory_strategy"]["preallocate"]:
                    if memory_allocated < 0.5:
                        torch.cuda.empty_cache()
                        logger.debug("Memory cache emptied (auto mode - preallocation)")
                
                if memory_allocated > self.cuda_config["memory_strategy"]["empty_cache_threshold"]:
                    torch.cuda.empty_cache()
                    logger.info(f"Memory cache emptied (auto mode - threshold {memory_allocated:.2%})")
            
            # Controllo frammentazione
            fragmentation = self._check_memory_fragmentation()
            if fragmentation > self.defrag_threshold:
                logger.info(f"High memory fragmentation detected: {fragmentation:.2%}")
                self._perform_defragmentation()
            
            # Garbage collection periodico
            if self.periodic_gc:
                current_time = time.time()
                if current_time - self.last_gc_time > self.gc_interval:
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.last_gc_time = current_time
                    logger.debug("Periodic garbage collection performed")
            
            # Forza rilascio memoria se critico
            if memory_allocated > self.cuda_config["memory_strategy"]["force_release_threshold"]:
                torch.cuda.empty_cache()
                gc.collect()
                logger.warning(f"Forced memory release (critical {memory_allocated:.2%})")
                
        except Exception as e:
            logger.error(f"Error in CUDA memory management: {str(e)}")
