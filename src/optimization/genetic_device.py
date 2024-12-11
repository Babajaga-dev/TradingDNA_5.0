import logging
import os
import psutil
import torch
import traceback
from typing import List

logger = logging.getLogger(__name__)

class DeviceManager:
    def __init__(self, config):
        self.use_gpu = config.get("genetic.optimizer.use_gpu", False)
        self.precision = config.get("genetic.optimizer.device_config.precision", "float32")
        self.dtype = torch.float16 if self.precision == "float16" else torch.float32
        
        try:
            if self.use_gpu and torch.cuda.is_available():
                self._setup_gpu(config)
            else:
                self._setup_cpu(config)
                
        except Exception as e:
            logger.error(f"Error during device setup: {str(e)}")
            logger.error(traceback.format_exc())
            self._setup_cpu(config)

    def _setup_gpu(self, config):
        self.num_gpus = torch.cuda.device_count()
        self.devices = [torch.device(f"cuda:{i}") for i in range(self.num_gpus)]
        logger.info(f"Using {self.num_gpus} CUDA devices")
        
        # Setup memoria GPU
        self.memory_reserve = config.get("genetic.optimizer.device_config.memory_reserve", 2048)
        self.max_batch_size = config.get("genetic.optimizer.device_config.max_batch_size", 1024)
        
        # Mixed precision
        self.mixed_precision = config.get("genetic.optimizer.device_config.mixed_precision", False)
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            
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
        self.cuda_config = {
            "allow_tf32": config.get("genetic.optimizer.cuda_config.allow_tf32", True),
            "benchmark": config.get("genetic.optimizer.cuda_config.benchmark", True),
            "deterministic": config.get("genetic.optimizer.cuda_config.deterministic", False),
            "memory_strategy": {
                "preallocate": config.get("genetic.optimizer.cuda_config.memory_strategy.preallocate", True),
                "empty_cache_threshold": config.get("genetic.optimizer.cuda_config.memory_strategy.empty_cache_threshold", 0.9),
                "force_release_threshold": config.get("genetic.optimizer.cuda_config.memory_strategy.force_release_threshold", 0.95)
            },
            "compute_capability": config.get("genetic.optimizer.cuda_config.compute_capability", "6.1"),
            "optimization_level": config.get("genetic.optimizer.cuda_config.optimization_level", 3)
        }
        
        if self.cuda_config["allow_tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        if self.cuda_config["benchmark"]:
            torch.backends.cudnn.benchmark = True
        
        torch.backends.cudnn.deterministic = self.cuda_config["deterministic"]

    def _setup_linux_specific(self, config):
        if os.name == 'posix':
            linux_config = config.get("genetic.optimizer.linux_specific", {})
            if linux_config.get("process_affinity", False):
                try:
                    process = psutil.Process()
                    process.cpu_affinity([0])
                except Exception as e:
                    logger.warning(f"Could not set process affinity: {str(e)}")
                    
            if linux_config.get("shared_memory", False):
                try:
                    torch.multiprocessing.set_sharing_strategy('file_system')
                except Exception as e:
                    logger.warning(f"Could not set shared memory strategy: {str(e)}")

    def _log_gpu_info(self):
        logger.info(f"CUDA enabled: {torch.cuda.is_available()}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"CUDA compute capability: {self.cuda_config['compute_capability']}")
        logger.info(f"CUDA optimization level: {self.cuda_config['optimization_level']}")
        
        for i in range(self.num_gpus):
            try:
                device_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                logger.info(f"GPU {i}: {device_name}")
                logger.info(f"Memory allocated: {memory_allocated:.2f} GB")
            except Exception as e:
                logger.warning(f"Could not get info for GPU {i}: {str(e)}")

    def manage_cuda_memory(self) -> None:
        if not self.use_gpu:
            return
            
        try:
            memory_allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
            if self.cuda_config["memory_strategy"]["preallocate"]:
                if memory_allocated < 0.5:
                    torch.cuda.empty_cache()
                    
            if memory_allocated > self.cuda_config["memory_strategy"]["empty_cache_threshold"]:
                torch.cuda.empty_cache()
                
            if memory_allocated > self.cuda_config["memory_strategy"]["force_release_threshold"]:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in CUDA memory management: {str(e)}")
