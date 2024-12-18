import unittest
import torch
import numpy as np
from src.models.simulator_processor import SimulationProcessor
from src.models.simulator_device import SimulatorDevice
from src.utils.config import config

class TestSimulationProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup iniziale che viene eseguito una volta sola prima di tutti i test"""
        # Carica la configurazione di test
        try:
            config.load_config("config_gpu_arc.yaml")
        except FileNotFoundError:
            # Fallback su configurazione minima se il file non esiste
            test_config = {
                "simulator": {
                    "initial_capital": 10000,
                    "metrics": {
                        "element_size_bytes": 32
                    }
                },
                "trading": {
                    "position": {
                        "size_pct": 40,
                        "stop_loss_pct": 1.5,
                        "take_profit_pct": 3.0
                    }
                },
                "genetic": {
                    "optimizer": {
                        "gpu_backend": "cpu",
                        "use_gpu": False,
                        "device_config": {
                            "precision": "float32"
                        }
                    },
                    "batch_processing": {
                        "enabled": True,
                        "adaptive_batching": True,
                        "min_batch_size": 32,
                        "max_batch_size": 128,
                        "memory_limit": 1024,
                        "prefetch_factor": 1,
                        "overlap_transfers": False
                    }
                }
            }
            config._config = test_config

    def setUp(self):
        """Setup che viene eseguito prima di ogni singolo test"""
        self.device_manager = SimulatorDevice(config)
        self.processor = SimulationProcessor(self.device_manager, config)
        
        # Setup test data
        self.size = 1000
        self.prices = torch.linspace(100, 200, self.size, device=self.device_manager.device)
        self.initial_capital = 10000.0

    def test_simple_entry_signal(self):
        """Test che una singola condizione di entrata generi almeno un trade"""
        # Crea un singolo segnale di entrata
        entry_conditions = torch.zeros(self.size, dtype=torch.bool, device=self.device_manager.device)
        entry_conditions[100] = True  # Segnale di entrata al timestep 100
        
        results = self.processor.run_simulation(
            prices=self.prices,
            entry_conditions=entry_conditions,
            initial_capital=self.initial_capital
        )
        
        # Verifica che almeno un trade sia stato eseguito
        total_trades = torch.sum(results["pnl"] != 0).item()
        self.assertGreater(total_trades, 0, "Nessun trade eseguito con segnale di entrata valido")

    def test_position_size(self):
        """Test che la position size sia correttamente calcolata"""
        entry_conditions = torch.zeros(self.size, dtype=torch.bool, device=self.device_manager.device)
        entry_conditions[100] = True
        
        results = self.processor.run_simulation(
            prices=self.prices,
            entry_conditions=entry_conditions,
            initial_capital=self.initial_capital
        )
        
        # Verifica position size
        max_position_size = torch.max(results["position_sizes"]).item()
        expected_size = self.processor.position_size_pct
        self.assertAlmostEqual(max_position_size, expected_size, places=2,
                             msg="Position size non corretta")

    def test_stop_loss(self):
        """Test che lo stop loss funzioni correttamente"""
        # Crea prezzi che triggherano uno stop loss
        prices = torch.ones(self.size, device=self.device_manager.device) * 100
        prices[200:] = 98  # -2% drop
        
        entry_conditions = torch.zeros(self.size, dtype=torch.bool, device=self.device_manager.device)
        entry_conditions[100] = True
        
        results = self.processor.run_simulation(
            prices=prices,
            entry_conditions=entry_conditions,
            initial_capital=self.initial_capital
        )
        
        # Verifica che la posizione sia stata chiusa
        position_closed = not results["position_active"][-1].item()
        self.assertTrue(position_closed, "Stop loss non ha chiuso la posizione")

    def test_multiple_positions(self):
        """Test che gestisca correttamente più posizioni"""
        # Crea più segnali di entrata
        entry_conditions = torch.zeros(self.size, dtype=torch.bool, device=self.device_manager.device)
        entry_conditions[100] = True
        entry_conditions[200] = True
        entry_conditions[300] = True
        
        results = self.processor.run_simulation(
            prices=self.prices,
            entry_conditions=entry_conditions,
            initial_capital=self.initial_capital
        )
        
        # Verifica numero massimo di posizioni contemporanee
        max_active = torch.max(torch.sum(results["position_active"])).item()
        self.assertLessEqual(max_active, self.processor.max_positions,
                           "Superato il numero massimo di posizioni consentite")
        
        # Verifica che almeno una posizione sia stata aperta
        total_trades = torch.sum(results["pnl"] != 0).item()
        self.assertGreater(total_trades, 0,
                          "Nessuna posizione aperta con segnali multipli")

    def test_take_profit(self):
        """Test che il take profit funzioni correttamente"""
        # Crea prezzi che triggherano un take profit
        prices = torch.ones(self.size, device=self.device_manager.device) * 100
        prices[200:] = 103  # +3% gain
        
        entry_conditions = torch.zeros(self.size, dtype=torch.bool, device=self.device_manager.device)
        entry_conditions[100] = True
        
        results = self.processor.run_simulation(
            prices=prices,
            entry_conditions=entry_conditions,
            initial_capital=self.initial_capital
        )
        
        # Verifica che la posizione sia stata chiusa in profitto
        position_closed = not results["position_active"][-1].item()
        final_pnl = (results["equity"][-1] - self.initial_capital).item()
        
        self.assertTrue(position_closed, "Take profit non ha chiuso la posizione")
        self.assertGreater(final_pnl, 0, "Take profit non ha generato profitto")

    def test_equity_update(self):
        """Test che l'equity venga aggiornata correttamente"""
        # Crea un trade profittevole
        prices = torch.ones(self.size, device=self.device_manager.device) * 100
        prices[200:] = 103
        
        entry_conditions = torch.zeros(self.size, dtype=torch.bool, device=self.device_manager.device)
        entry_conditions[100] = True
        
        results = self.processor.run_simulation(
            prices=prices,
            entry_conditions=entry_conditions,
            initial_capital=self.initial_capital
        )
        
        # Verifica che l'equity sia monotona crescente
        equity_diffs = torch.diff(results["equity"])
        all_non_negative = torch.all(equity_diffs >= 0).item()
        
        self.assertTrue(all_non_negative, "L'equity non è monotona crescente")

if __name__ == '__main__':
    unittest.main()