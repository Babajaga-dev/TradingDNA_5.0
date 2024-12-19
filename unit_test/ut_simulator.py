import unittest
import torch
import numpy as np
import logging
from src.models.simulator_processor import SimulationProcessor
from src.models.simulator_device import SimulatorDevice
from src.models.position_manager import PositionManager
from src.models.risk_manager import RiskManager
from src.utils.config import config

# Configura root logger
logging.basicConfig(
    level=logging.WARNING,  # Root logger a WARNING per sopprimere la maggior parte dei log
    format='%(message)s'
)

# Configura logger specifico per simulator_processor
sim_logger = logging.getLogger('src.models.simulator_processor')
sim_logger.setLevel(logging.DEBUG)
sim_logger.propagate = False  # Previene la propagazione al root logger

# Configura logger per simulator_device
device_logger = logging.getLogger('src.models.simulator_device')
device_logger.setLevel(logging.WARNING)  # Riduce i log del device
device_logger.propagate = False

# Handler comune per i log specifici
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(message)s'))

# Aggiungi handler ai logger specifici
sim_logger.addHandler(console_handler)
device_logger.addHandler(console_handler)

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
                        "size_pct": 0.4,
                        "stop_loss_pct": 0.015,
                        "take_profit_pct": 0.03
                    },
                    "risk_management": {
                        "max_drawdown_pct": 0.15  # già in decimale
                    }
                },
                "genetic": {
                    "optimizer": {
                        "gpu_backend": "cpu",
                        "use_gpu": False,
                        "device_config": {
                            "precision": "float32"
                        }
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

    def test_position_manager(self):
        """Test delle funzionalità del PositionManager"""
        position_manager = PositionManager(config)
        
        # Test calcolo max positions
        expected_max = int(1 / position_manager.position_size_pct)
        self.assertEqual(position_manager.max_positions, expected_max,
                        "Calcolo errato del numero massimo di posizioni")
        
        # Test inizializzazione tensori
        tensors = position_manager.initialize_position_tensors(
            size=100,
            device=self.device_manager.device,
            dtype=torch.float32
        )
        self.assertIn("active_positions", tensors, "Manca il tensore active_positions")
        self.assertIn("entry_prices", tensors, "Manca il tensore entry_prices")
        self.assertIn("position_sizes", tensors, "Manca il tensore position_sizes")
        
        # Test condizioni di chiusura
        price_changes = torch.tensor([[-2.0, 0.0], [3.5, 1.0]], device=self.device_manager.device)
        active_positions = torch.tensor([[True, False], [True, True]], device=self.device_manager.device)
        close_mask = position_manager.check_close_conditions(
            price_changes=price_changes,
            active_positions=active_positions,
            max_drawdown_hit=False
        )
        self.assertTrue(close_mask[0,0], "Stop loss non rilevato")
        self.assertTrue(close_mask[1,0], "Take profit non rilevato")

    def test_risk_manager(self):
        """Test delle funzionalità del RiskManager"""
        risk_manager = RiskManager(config)
        
        # Test max drawdown
        self.assertTrue(
            risk_manager.check_max_drawdown(
                current_equity=8000,  # -20%
                initial_capital=10000
            ),
            "Max drawdown non rilevato"
        )
        
        self.assertFalse(
            risk_manager.check_max_drawdown(
                current_equity=9000,  # -10%
                initial_capital=10000
            ),
            "Max drawdown rilevato erroneamente"
        )
        
        # Test validazione position size
        self.assertTrue(
            risk_manager.validate_position_size(0.05, 10000),
            "Position size valida non accettata"
        )
        
        self.assertFalse(
            risk_manager.validate_position_size(0.0001, 100),
            "Position size troppo piccola accettata"
        )

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
        expected_size = self.processor.position_manager.position_size_pct
        self.assertAlmostEqual(max_position_size, expected_size, places=2,
                             msg="Position size non corretta")

    def test_position_closure(self):
        """Test dettagliato della logica di chiusura delle posizioni"""
        # Crea una sequenza di prezzi che dovrebbe triggerare uno stop loss
        prices = torch.tensor([
            100.0,  # t0: prezzo iniziale
            100.0,  # t1: entry price
            99.0,   # t2: -1.0%
            98.0,   # t3: -2.0% (dovrebbe triggerare stop loss)
            97.0    # t4: -3.0%
        ], device=self.device_manager.device)
        
        # Segnale di entrata al tempo t1
        entry_conditions = torch.tensor([
            False,  # t0
            True,   # t1: apri posizione
            False,  # t2
            False,  # t3
            False   # t4
        ], device=self.device_manager.device)
        
        results = self.processor.run_simulation(
            prices=prices,
            entry_conditions=entry_conditions,
            initial_capital=self.initial_capital
        )
        
        # Verifica che la posizione sia stata aperta
        self.assertTrue(results["position_active"][1].any(), 
                       "La posizione non è stata aperta al tempo t1")
        
        # Verifica che la posizione sia stata chiusa
        self.assertFalse(results["position_active"][3].any(), 
                        "La posizione non è stata chiusa quando il prezzo ha superato lo stop loss")
        
        # Verifica il PnL
        pnl = results["pnl"][3].item()
        expected_pnl = -self.initial_capital * self.processor.position_manager.position_size_pct * 0.02  # -2% loss
        self.assertAlmostEqual(pnl, expected_pnl, places=2,
                             msg=f"PnL non corretto. Atteso: {expected_pnl:.2f}, Ottenuto: {pnl:.2f}")

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
        
        # Verifica numero massimo di posizioni contemporanee per timestep
        max_active = torch.max(torch.sum(results["position_active"], dim=1)).item()
        self.assertLessEqual(max_active, self.processor.position_manager.max_positions,
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
        position_closed = not results["position_active"][-1].any()
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
