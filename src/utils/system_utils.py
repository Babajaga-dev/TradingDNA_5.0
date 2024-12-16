# src/utils/system_utils.py
import os
import platform
import logging
import psutil
import torch
import intel_extension_for_pytorch as ipex
import torch.multiprocessing as mp
from pathlib import Path
from typing import Optional, Dict, List, Any, TypedDict, Union, cast
from contextlib import contextmanager
import warnings
from .system_helper import system_helper

logger = logging.getLogger(__name__)

class PathConfig(TypedDict):
    data_dir: Path
    config_dir: Path
    cache_dir: Path
    base_dir: Path

class PathManager:
    def __init__(self, base_dir: Optional[str] = None):
        """
        Gestisce i percorsi del sistema in modo cross-platform.
        
        Args:
            base_dir: Directory base opzionale. Se non specificata, usa la directory corrente.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.is_linux = system_helper.is_linux

    def get_data_dir(self) -> Path:
        """Restituisce il percorso della directory dei dati"""
        try:
            if self.is_linux:
                xdg_data = os.environ.get('XDG_DATA_HOME')
                if xdg_data:
                    return Path(xdg_data) / "trading_system"
                return Path.home() / ".local/share/trading_system"
            return self.base_dir / "data"
        except Exception as e:
            logger.error(f"Error getting data directory: {e}")
            return self.base_dir / "data"  # Fallback sicuro

    def get_config_dir(self) -> Path:
        """Restituisce il percorso della directory di configurazione"""
        try:
            if self.is_linux:
                xdg_config = os.environ.get('XDG_CONFIG_HOME')
                if xdg_config:
                    return Path(xdg_config) / "trading_system"
                return Path.home() / ".config/trading_system"
            return self.base_dir / "config"
        except Exception as e:
            logger.error(f"Error getting config directory: {e}")
            return self.base_dir / "config"  # Fallback sicuro

    def get_cache_dir(self) -> Path:
        """Restituisce il percorso della directory di cache"""
        try:
            if self.is_linux:
                xdg_cache = os.environ.get('XDG_CACHE_HOME')
                if xdg_cache:
                    return Path(xdg_cache) / "trading_system"
                return Path.home() / ".cache/trading_system"
            return self.base_dir / "cache"
        except Exception as e:
            logger.error(f"Error getting cache directory: {e}")
            return self.base_dir / "cache"  # Fallback sicuro

    def ensure_directories(self) -> List[Path]:
        """Crea tutte le directory necessarie se non esistono"""
        dirs = [
            self.get_data_dir(),
            self.get_config_dir(),
            self.get_cache_dir()
        ]
        
        created_dirs = []
        for directory in dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                created_dirs.append(directory)
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {e}")
        
        return created_dirs

    def get_absolute_path(self, relative_path: Union[str, Path]) -> Path:
        """
        Converte un percorso relativo in assoluto
        
        Args:
            relative_path: Percorso relativo da convertire
            
        Returns:
            Path assoluto
        """
        try:
            path = Path(relative_path)
            if path.is_absolute():
                return path
            return self.base_dir / path
        except Exception as e:
            logger.error(f"Error converting path {relative_path}: {e}")
            return self.base_dir  # Fallback sicuro

    def get_paths_config(self) -> PathConfig:
        """
        Restituisce la configurazione completa dei percorsi
        
        Returns:
            Dict con tutti i percorsi configurati
        """
        return {
            'data_dir': self.get_data_dir(),
            'config_dir': self.get_config_dir(),
            'cache_dir': self.get_cache_dir(),
            'base_dir': self.base_dir
        }

def setup_training_environment() -> None:
    """Configura l'ambiente per il training"""
    try:
        # Configura logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Configura multiprocessing
        if system_helper.is_linux:
            mp.set_start_method('spawn', force=True)
        
        # Configura PyTorch
        torch.set_default_dtype(torch.float32)
        
        # Configura CUDA se disponibile
        if system_helper.cuda_available:
            torch.cuda.amp.autocast(enabled=True)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Configura XPU se disponibile
        if torch.xpu.is_available():
            ipex.optimize(dtype=torch.float16)
        
        # Ottimizza sistema
        system_helper.optimize_thread_settings()
        system_helper.setup_cuda_cache()
        
        # Log configurazione
        logger.info("Training environment configured successfully")
        
    except Exception as e:
        logger.error(f"Error setting up training environment: {e}")
        raise

def get_optimal_batch_size(sample_size_bytes: int) -> int:
    """
    Calcola la dimensione ottimale del batch
    
    Args:
        sample_size_bytes: Dimensione in bytes di un singolo campione
        
    Returns:
        Dimensione ottimale del batch
    """
    try:
        return system_helper.get_optimal_batch_size(sample_size_bytes)
    except Exception as e:
        logger.error(f"Error calculating optimal batch size: {e}")
        return 32  # Fallback sicuro

def get_optimal_worker_count() -> int:
    """
    Calcola il numero ottimale di workers per DataLoader
    
    Returns:
        Numero ottimale di workers
    """
    try:
        cpu_count = psutil.cpu_count(logical=False) or 1
        return max(1, min(cpu_count - 1, 8))  # Massimo 8 workers, lascia 1 core libero
    except Exception as e:
        logger.error(f"Error calculating optimal worker count: {e}")
        return 1  # Fallback sicuro

@contextmanager
def safe_gpu_memory():
    """Context manager per gestire la memoria GPU in modo sicuro"""
    try:
        # Gestione CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                yield
            torch.cuda.empty_cache()
        # Gestione XPU
        elif torch.xpu.is_available():
            torch.xpu.empty_cache()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                yield
            torch.xpu.empty_cache()
        else:
            yield
    except Exception as e:
        logger.error(f"Error in GPU memory management: {e}")
        raise
