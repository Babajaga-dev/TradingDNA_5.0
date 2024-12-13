import yaml
import logging
from typing import Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    _instance = None
    _config = None
    _config_path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        pass  # Rimuovo il caricamento automatico

    def load_config(self, config_path: str = "config.yaml"):
        try:
            self._config_path = config_path
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")
                
            logger.debug(f"Caricamento configurazione da {config_path}")
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
                
            logger.debug("Configurazione caricata con successo")
            logger.debug(f"Configurazione completa: {self._config}")
                
        except Exception as e:
            logger.error(f"Errore nel caricamento della configurazione: {str(e)}")
            raise

    def get(self, path: str, default: Any = None) -> Any:
        if self._config is None:
            raise RuntimeError("Configurazione non ancora caricata. Chiamare load_config() prima di get()")
            
        try:
            value = self._config
            for key in path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Chiave {path} non trovata, uso valore default: {default}")
            return default

    def get_all(self) -> Dict:
        if self._config is None:
            raise RuntimeError("Configurazione non ancora caricata. Chiamare load_config() prima di get_all()")
        return self._config

    def get_config_path(self) -> str:
        return self._config_path

# Istanza singleton
config = Config()
