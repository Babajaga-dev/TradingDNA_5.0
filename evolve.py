# evolve.py
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict

from src.utils.data import load_and_prepare_data
from src.models.common import TimeFrame
from src.models.simulator import TradingSimulator
from src.optimization.genetic import ParallelGeneticOptimizer
from src.utils.config import config

def run_evolution(data_file: str,
                 generations: int = None,
                 save_path: str = "evolution_state.pkl",
                 population_size: int = None) -> None:
    """Esegue l'evoluzione genetica della popolazione"""
    
    print("\nINIZIALIZZAZIONE EVOLUZIONE")
    print("="*50)
    
    # Override configurazione se specificato
    if generations:
        config._config['genetic']['generations'] = generations
    if population_size:
        config._config['genetic']['population_size'] = population_size
        
    # Carica e prepara dati
    print("Caricamento dati di mercato...")
    market_data = load_and_prepare_data(data_file)
    
    # Setup simulatore
    print("Inizializzazione simulatore...")
    simulator = TradingSimulator()
    simulator.add_market_data(TimeFrame.M1, market_data['1m'])
    
    # Esegui ottimizzazione
    print("\nAvvio ottimizzazione genetica...")
    optimizer = ParallelGeneticOptimizer()
    best_gene, stats = optimizer.optimize(simulator)
    
    # Prepara stato da salvare
    evolution_state = {
        'timestamp': datetime.now(),
        'best_gene': best_gene,
        'stats': stats,
        'config': config.get_all(),
        'generation_stats': optimizer.generation_stats,
        'data_info': {
            'filename': Path(data_file).name,
            'timeframe': '1m',
            'start_date': market_data['1m']['timestamp'].min(),
            'end_date': market_data['1m']['timestamp'].max()
        }
    }
    
    # Salva stato
    print(f"\nSalvataggio stato evoluzione in {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(evolution_state, f)
        
    print("\nEvoluzione completata!")
    print(f"Miglior fitness: {stats['best_fitness']:.4f}")