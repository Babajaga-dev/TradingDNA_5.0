# src/utils/system_helper.py
import os
import sys
import platform
import subprocess
import psutil
import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Union, Any, TypedDict, cast
from contextlib import contextmanager
import torch
import intel_extension_for_pytorch as ipex
import numpy as np

try:
    import win32process
    import win32api
    WINDOWS_API_AVAILABLE = True
except ImportError:
    WINDOWS_API_AVAILABLE = False

logger = logging.getLogger(__name__)

class GPUInfo(TypedDict):
    index: int
    name: str
    total_memory: int
    free_memory: int
    used_memory: int
    compute_capability: str

class SystemHelper:
    def __init__(self):
        self.os_name = platform.system().lower()
        self.is_linux = self.os_name == 'linux'
        self.is_windows = self.os_name == 'windows'
        self.python_version = sys.version_info[:3]
        self.cuda_available = torch.cuda.is_available()
        self.xpu_available = torch.xpu.is_available()

    def get_memory_info(self) -> Dict[str, int]:
        """Ottiene informazioni sulla memoria del sistema"""
        try:
            vm = psutil.virtual_memory()
            memory_info = {
                'total': vm.total,
                'available': vm.available,
                'used': vm.used,
                'free': vm.free
            }
            
            if self.is_linux:
                memory_info.update({
                    'cached': vm.cached,
                    'buffers': vm.buffers
                })
                
            return {k: v for k, v in memory_info.items() if v is not None}
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {'error': 'Failed to get memory info'}

    def get_gpu_info(self) -> List[GPUInfo]:
        """Ottiene informazioni sulle GPU disponibili"""
        gpu_info: List[GPUInfo] = []
        
        # Info NVIDIA GPU
        if self.cuda_available:
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    gpu_info.append({
                        'index': i,
                        'name': props.name,
                        'total_memory': total_mem,
                        'free_memory': free_mem,
                        'used_memory': total_mem - free_mem,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
            except Exception as e:
                logger.error(f"Error getting NVIDIA GPU info: {e}")
                
        # Info Intel GPU
        if self.xpu_available:
            try:
                for i in range(torch.xpu.device_count()):
                    free_mem, total_mem = torch.xpu.mem_get_info(i)
                    gpu_info.append({
                        'index': i,
                        'name': "Intel Arc GPU",
                        'total_memory': total_mem,
                        'free_memory': free_mem,
                        'used_memory': total_mem - free_mem,
                        'compute_capability': "1.0"  # Intel Arc non ha compute capability come NVIDIA
                    })
            except Exception as e:
                logger.error(f"Error getting Intel GPU info: {e}")
                
        return gpu_info

    def setup_cuda_cache(self, cache_dir: Optional[Path] = None) -> None:
        """Configura la cache GPU"""
        if not (self.cuda_available or self.xpu_available):
            return

        try:
            if cache_dir is None:
                if self.is_linux:
                    cache_dir = Path.home() / '.cache' / 'torch'
                else:
                    cache_dir = Path.home() / 'AppData' / 'Local' / 'torch'

            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ['TORCH_HOME'] = str(cache_dir)
            
            if self.cuda_available:
                torch.cuda.empty_cache()
            if self.xpu_available:
                torch.xpu.empty_cache()
        except Exception as e:
            logger.error(f"Error setting up GPU cache: {e}")

    def optimize_thread_settings(self) -> None:
        """Ottimizza le impostazioni dei thread"""
        try:
            cpu_count = psutil.cpu_count(logical=False) or 1
            if self.is_linux:
                # Usa OpenMP per controllo thread
                os.environ['OMP_NUM_THREADS'] = str(max(1, cpu_count - 1))
                os.environ['MKL_NUM_THREADS'] = str(max(1, cpu_count - 1))
            
            # Imposta thread PyTorch
            torch.set_num_threads(max(1, cpu_count - 1))
            torch.set_num_interop_threads(max(1, cpu_count // 2))
        except Exception as e:
            logger.error(f"Error optimizing thread settings: {e}")

    @contextmanager
    def temp_gpu_memory_limit(self, limit_mb: int):
        """Context manager per limitare temporaneamente la memoria GPU"""
        if not (self.cuda_available or self.xpu_available):
            yield
            return

        old_limits = []
        try:
            # Salva limiti CUDA
            if self.cuda_available:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        old_limit = torch.cuda.get_memory_allocation_limit()
                        old_limits.append(('cuda', i, old_limit))
                        torch.cuda.set_per_process_memory_fraction(limit_mb / 1024)
                        
            # Salva limiti XPU
            if self.xpu_available:
                for i in range(torch.xpu.device_count()):
                    old_limit = torch.xpu.get_memory_allocation_limit()
                    old_limits.append(('xpu', i, old_limit))
                    # XPU non ha un metodo diretto per limitare la memoria, 
                    # ma possiamo usare ipex per ottimizzare l'uso della memoria
                    ipex.optimize_for_memory()
                    
            yield
        except Exception as e:
            logger.error(f"Error setting GPU memory limit: {e}")
            raise
        finally:
            # Ripristina limiti
            for backend, i, old_limit in old_limits:
                try:
                    if backend == 'cuda':
                        with torch.cuda.device(i):
                            torch.cuda.set_memory_allocation_limit(old_limit)
                except Exception as e:
                    logger.error(f"Error restoring GPU memory limit: {e}")

    def setup_process_priority(self, high_priority: bool = False) -> None:
        """Imposta la priorità del processo"""
        try:
            process = psutil.Process()
            if self.is_linux:
                # nice va da -20 (highest) a 19 (lowest)
                niceness = -10 if high_priority else 10
                os.nice(niceness)
            elif self.is_windows and WINDOWS_API_AVAILABLE:
                # Windows priority
                priority = psutil.HIGH_PRIORITY_CLASS if high_priority else psutil.BELOW_NORMAL_PRIORITY_CLASS
                process.nice(priority)
        except Exception as e:
            logger.warning(f"Could not set process priority: {e}")

    def get_optimal_batch_size(self, sample_size_bytes: int) -> int:
        """Calcola la dimensione ottimale del batch in base alla memoria disponibile"""
        try:
            if self.cuda_available:
                # Usa memoria CUDA GPU
                free_mem = min(
                    torch.cuda.mem_get_info(i)[0] 
                    for i in range(torch.cuda.device_count())
                )
                # Riserva 20% della memoria libera
                usable_mem = int(free_mem * 0.8)
            elif self.xpu_available:
                # Usa memoria XPU GPU
                free_mem = min(
                    torch.xpu.mem_get_info(i)[0]
                    for i in range(torch.xpu.device_count())
                )
                # Riserva 20% della memoria libera
                usable_mem = int(free_mem * 0.8)
            else:
                # Usa memoria RAM
                vm = psutil.virtual_memory()
                # Usa max 50% della memoria disponibile
                usable_mem = int(vm.available * 0.5)

            max_batch = usable_mem // max(1, sample_size_bytes)
            # Limita a valori ragionevoli
            return max(1, min(8192, max_batch))
        except Exception as e:
            logger.error(f"Error calculating optimal batch size: {e}")
            return 32  # Default sicuro

    def estimate_memory_usage(self, shape: tuple, dtype: np.dtype) -> int:
        """Stima l'utilizzo di memoria per un array"""
        try:
            return int(np.prod(shape) * dtype.itemsize)
        except Exception as e:
            logger.error(f"Error estimating memory usage: {e}")
            return 0

    def clear_gpu_memory(self) -> None:
        """Libera la memoria GPU"""
        try:
            if self.cuda_available:
                torch.cuda.empty_cache()
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.memory._dump_snapshot(flush_gpu_cache=True)
                        
            if self.xpu_available:
                torch.xpu.empty_cache()
                # XPU non ha un metodo equivalente a _dump_snapshot
                
        except Exception as e:
            logger.error(f"Error clearing GPU memory: {e}")

    def setup_environment(self) -> None:
        """Configura l'ambiente di sistema"""
        try:
            if self.is_linux:
                self._setup_linux_environment()
            elif self.is_windows:
                self._setup_windows_environment()

            # Configurazione comune
            self.optimize_thread_settings()
            self.setup_cuda_cache()
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")

    def _setup_linux_environment(self) -> None:
        """Configurazione specifica per Linux"""
        try:
            # Imposta variabili ambiente
            os.environ['TMPDIR'] = '/tmp'
            
            # Configura numa se disponibile
            if shutil.which('numactl'):
                try:
                    subprocess.run(['numactl', '--interleave=all', sys.executable], check=True)
                except subprocess.SubprocessError as e:
                    logger.warning(f"Could not set NUMA policy: {e}")

            # Imposta scheduler I/O
            try:
                with open('/proc/self/oom_score_adj', 'w') as f:
                    f.write('1000\n')
            except IOError as e:
                logger.warning(f"Could not adjust OOM score: {e}")
        except Exception as e:
            logger.error(f"Error setting up Linux environment: {e}")

    def _setup_windows_environment(self) -> None:
        """Configurazione specifica per Windows"""
        try:
            # Imposta variabili ambiente
            os.environ['TEMP'] = os.path.expandvars('%LOCALAPPDATA%\\Temp')
            
            # Disabilita priority boost
            if WINDOWS_API_AVAILABLE:
                try:
                    handle = win32api.GetCurrentProcess()
                    win32process.SetPriorityBoost(handle, False)
                except Exception as e:
                    logger.warning(f"Could not set priority boost: {e}")
        except Exception as e:
            logger.error(f"Error setting up Windows environment: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Ottiene lo stato completo del sistema"""
        try:
            status: Dict[str, Any] = {
                'os': self.os_name,
                'python_version': '.'.join(map(str, self.python_version)),
                'cpu': {
                    'physical_cores': psutil.cpu_count(logical=False),
                    'total_cores': psutil.cpu_count(logical=True),
                }
            }

            # Aggiungi info CPU frequenza se disponibile
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                status['cpu']['frequency'] = cpu_freq._asdict()

            # Aggiungi load average solo su Linux
            if self.is_linux:
                status['cpu']['load'] = os.getloadavg()

            # Info memoria
            status['memory'] = self.get_memory_info()

            # Info GPU
            if self.cuda_available or self.xpu_available:
                status['gpu'] = self.get_gpu_info()

            # Info disco
            disk_usage = psutil.disk_usage('/')
            status['disk'] = {'usage': disk_usage._asdict()}

            disk_io = psutil.disk_io_counters()
            if disk_io:
                status['disk']['io_counters'] = disk_io._asdict()

            return status
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

# Crea istanza singleton
system_helper = SystemHelper()
