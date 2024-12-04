import yaml
from typing import Dict, Any
from pathlib import Path

class Config:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    def load_config(self, config_path: str = "config.yaml"):
        """Carica la configurazione dal file YAML"""
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Errore nel caricamento del file di configurazione: {str(e)}")

    def get(self, path: str, default: Any = None) -> Any:
        """
        Ottiene un valore dalla configurazione usando un path con dot notation
        Esempio: config.get("simulator.initial_capital")
        """
        try:
            value = self._config
            for key in path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_all(self) -> Dict:
        """Restituisce l'intera configurazione"""
        return self._config

# Singleton instance
config = Config()
