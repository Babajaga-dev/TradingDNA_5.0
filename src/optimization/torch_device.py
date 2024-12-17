import torch
import logging
import psutil
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DeviceConfig:
    """Configurazione di un dispositivo (CPU, CUDA o XPU)"""
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

            # Controlla Intel XPU
            try:
                import intel_extension_for_pytorch as ipex
                if torch.xpu.is_available():
                    for i in range(torch.xpu.device_count()):
                        try:
                            mem_free, mem_total = torch.xpu.mem_get_info(i)
                            self.devices.append(DeviceConfig(
                                device_type="xpu",
                                device_index=i,
                                name="Intel Arc GPU",
                                memory_total=mem_total,
                                memory_free=mem_free,
                                compute_capability=(1, 0)  # Intel Arc non ha compute capability come NVIDIA
                            ))
                        except Exception as e:
                            logger.error(f"Error detecting XPU {i}: {e}")
            except ImportError:
                pass

            # Controlla NVIDIA GPU
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
                if dev.device_type in ["cuda", "xpu"]:
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

    def _validate_compute_capability(self, device_cc: Tuple[int, int], required_cc: str) -> bool:
        """
        Valida che la compute capability del dispositivo soddisfi i requisiti
        
        Args:
            device_cc: Compute capability del dispositivo (major, minor)
            required_cc: Compute capability richiesta (formato "major.minor")
            
        Returns:
            True se la compute capability è compatibile
        """
        try:
            req_major, req_minor = map(int, required_cc.split("."))
            dev_major, dev_minor = device_cc
            
            if dev_major > req_major:
                return True
            elif dev_major == req_major:
                return dev_minor >= req_minor
            return False
        except:
            logger.warning(f"Invalid compute capability format: {required_cc}")
            return True

    def _apply_optimization_level(self, level: int, device_type: str) -> None:
        """
        Applica il livello di ottimizzazione per il dispositivo
        
        Args:
            level: Livello di ottimizzazione (0-3)
            device_type: Tipo di dispositivo ("cuda" o "xpu")
        """
        try:
            if not 0 <= level <= 3:
                logger.warning(f"Invalid optimization level: {level}. Using default (3)")
                level = 3
                
            if device_type == "cuda":
                if level >= 1:
                    torch.backends.cudnn.enabled = True
                if level >= 2:
                    torch.backends.cudnn.benchmark = True
                if level == 3:
                    # Abilita ottimizzazioni aggressive
                    if hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = True
                    if hasattr(torch, 'set_float32_matmul_precision'):
                        torch.set_float32_matmul_precision('high')
            elif device_type == "xpu":
                try:
                    import intel_extension_for_pytorch as ipex
                    if level >= 1:
                        ipex.enable_auto_mixed_precision(dtype='float16')
                    if level >= 2:
                        # Ottimizzazioni aggiuntive per XPU
                        ipex.optimize_for_inference()
                    if level == 3:
                        # Massime ottimizzazioni per XPU
                        ipex.optimize_for_training()
                except ImportError:
                    logger.warning("Intel Extension for PyTorch not found, skipping XPU optimizations")
                    
            logger.info(f"Applied {device_type.upper()} optimization level: {level}")
        except Exception as e:
            logger.error(f"Error applying optimization level: {e}")

    def get_best_device(self, config) -> torch.device:
        """
        Seleziona il miglior dispositivo disponibile
        
        Args:
            config: Configurazione
            
        Returns:
            Device PyTorch ottimale
        """
        try:
            use_gpu = config.get("genetic.optimizer.use_gpu", False)
            gpu_backend = config.get("genetic.optimizer.gpu_backend", "auto")
            
            if not use_gpu:
                logger.info("Using CPU device (GPU disabled in config)")
                return torch.device("cpu")
                
            # Cerca dispositivi GPU disponibili
            xpu_devices = [d for d in self.devices if d.device_type == "xpu"]
            cuda_devices = [d for d in self.devices if d.device_type == "cuda"]
            
            # Selezione automatica backend
            if gpu_backend == "auto":
                if xpu_devices:
                    gpu_backend = "arc"
                elif cuda_devices:
                    gpu_backend = "cuda"
                else:
                    gpu_backend = "cpu"
            
            # Seleziona dispositivo in base al backend
            if gpu_backend == "arc" and xpu_devices:
                # Seleziona XPU con più memoria libera
                best_gpu = max(xpu_devices, key=lambda x: x.memory_free)
                logger.info(f"Selected XPU device: {best_gpu.name}")
                return torch.device("xpu")
            elif gpu_backend == "cuda" and cuda_devices:
                # Verifica compute capability per CUDA
                required_cc = config.get("genetic.optimizer.cuda_config.compute_capability")
                if required_cc:
                    cuda_devices = [
                        d for d in cuda_devices 
                        if self._validate_compute_capability(d.compute_capability, required_cc)
                    ]
                    
                if not cuda_devices:
                    logger.warning("No GPU meets compute capability requirements, falling back to CPU")
                    return torch.device("cpu")
                
                # Seleziona GPU con più memoria libera
                best_gpu = max(cuda_devices, key=lambda x: x.memory_free)
                logger.info(f"Selected CUDA device: {best_gpu.name}")
                return torch.device(f"cuda:{best_gpu.device_index}")
            
            logger.info("No suitable GPU found, using CPU device")
            return torch.device("cpu")
                
        except Exception as e:
            logger.error(f"Error selecting device: {e}")
            return torch.device("cpu")

    def setup_device(self, device: torch.device, config) -> None:
        """
        Configura il dispositivo selezionato
        
        Args:
            device: Device PyTorch
            config: Configurazione
        """
        if device.type == "cpu":
            torch_threads = config.get("genetic.optimizer.torch_threads", None)
            if torch_threads:
                torch.set_num_threads(torch_threads)
                logger.info(f"Set CPU threads to {torch_threads}")
                
        elif device.type == "xpu":
            try:
                import intel_extension_for_pytorch as ipex
                # Configura XPU
                xpu_config = config.get("genetic.optimizer.device_config.arc", {})
                
                # Applica livello ottimizzazione
                opt_level = xpu_config.get("optimization_level", 3)
                self._apply_optimization_level(opt_level, "xpu")
                
                # Configura mixed precision
                if xpu_config.get("mixed_precision", True):
                    ipex.enable_auto_mixed_precision(dtype='float16')
                    logger.info("XPU mixed precision enabled")
                
                # Log configurazione finale
                logger.info("XPU configuration:")
                logger.info(f"- Optimization level: {opt_level}")
                logger.info(f"- Mixed precision: {xpu_config.get('mixed_precision', True)}")
            except ImportError:
                logger.warning("Intel Extension for PyTorch not found, using basic XPU configuration")
            
        else:  # cuda
            # Configura CUDA
            cuda_config = config.get("genetic.optimizer.cuda_config", {})
            
            # Applica livello ottimizzazione
            opt_level = cuda_config.get("optimization_level", 3)
            self._apply_optimization_level(opt_level, "cuda")
            
            # Configura TF32
            if hasattr(torch.backends.cuda, 'allow_tf32'):
                allow_tf32 = cuda_config.get("allow_tf32", True)
                torch.backends.cuda.allow_tf32 = allow_tf32
                torch.backends.cudnn.allow_tf32 = allow_tf32
                logger.info(f"TF32 support: {allow_tf32}")
            
            # Configura benchmark e deterministic
            torch.backends.cudnn.benchmark = cuda_config.get("benchmark", True)
            torch.backends.cudnn.deterministic = cuda_config.get("deterministic", False)
            
            # Log configurazione finale
            logger.info("CUDA configuration:")
            logger.info(f"- Optimization level: {opt_level}")
            logger.info(f"- Benchmark mode: {torch.backends.cudnn.benchmark}")
            logger.info(f"- Deterministic mode: {torch.backends.cudnn.deterministic}")
            if hasattr(torch.backends.cuda, 'allow_tf32'):
                logger.info(f"- TF32 enabled: {torch.backends.cuda.allow_tf32}")
