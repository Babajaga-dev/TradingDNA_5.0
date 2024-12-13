import torch
import logging
import psutil
from typing import List, Tuple, Optional
from dataclasses import dataclass

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

    def _apply_optimization_level(self, level: int) -> None:
        """
        Applica il livello di ottimizzazione CUDA
        
        Args:
            level: Livello di ottimizzazione (0-3)
        """
        try:
            if not 0 <= level <= 3:
                logger.warning(f"Invalid optimization level: {level}. Using default (3)")
                level = 3
                
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
                    
            logger.info(f"Applied CUDA optimization level: {level}")
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
            
            if use_gpu and len([d for d in self.devices if d.device_type == "cuda"]) > 0:
                # Verifica compute capability richiesta
                required_cc = config.get("genetic.optimizer.cuda_config.compute_capability")
                cuda_devices = [d for d in self.devices if d.device_type == "cuda"]
                
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
                logger.info(f"Selected GPU device: {best_gpu.name}")
                return torch.device(f"cuda:{best_gpu.device_index}")
            else:
                logger.info("Using CPU device")
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
        else:
            # Configura CUDA
            cuda_config = config.get("genetic.optimizer.cuda_config", {})
            
            # Applica livello ottimizzazione
            opt_level = cuda_config.get("optimization_level", 3)
            self._apply_optimization_level(opt_level)
            
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
