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
                # Seleziona GPU con più memoria libera
                best_gpu = max(
                    [d for d in self.devices if d.device_type == "cuda"],
                    key=lambda x: x.memory_free
                )
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
            torch.backends.cudnn.benchmark = config.get("genetic.optimizer.cuda_config.benchmark", True)
            torch.backends.cudnn.deterministic = config.get("genetic.optimizer.cuda_config.deterministic", False)
            logger.info("CUDA configuration applied")
