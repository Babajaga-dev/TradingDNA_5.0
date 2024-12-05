import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pickle
import json
from datetime import datetime
from typing import Optional

from src.utils.data import load_and_prepare_data
from src.models.common import TimeFrame
from src.models.simulator import TradingSimulator
from src.utils.config import config

def run_simulation(data_file: str,
                  evolution_state_path: str = "evolution_state.pkl",
                  output_file: Optional[str] = None) -> None:
    """Esegue simulazione trading con gene ottimizzato"""
    
    print("\nINIZIALIZZAZIONE SIMULAZIONE")
    print("="*50)
    
    # Carica stato evoluzione
    print(f"Caricamento stato da {evolution_state_path}")
    with open(evolution_state_path, 'rb') as f:
        evolution_state = pickle.load(f)
    
    best_gene = evolution_state['best_gene']
    evolution_timestamp = evolution_state['timestamp']
    
    print(f"\nStato evoluzione del: {evolution_timestamp}")
    print(f"Miglior fitness: {evolution_state['stats']['best_fitness']:.4f}")
    
    # Carica dati mercato
    print("\nCaricamento dati di mercato...")
    market_data = load_and_prepare_data(data_file)
    
    # Setup simulazione
    print("Inizializzazione simulatore...")
    simulator = TradingSimulator()
    simulator.add_market_data(TimeFrame.M1, market_data['1m'])
    
    # Esegui simulazione
    print("\nAvvio simulazione...")
    simulator.run_simulation(best_gene)
    metrics = simulator.get_performance_metrics()
    
    # Prepara risultati
    results = {
        'timestamp': datetime.now().isoformat(),
        'evolution_state': evolution_state_path,
        'evolution_timestamp': evolution_timestamp.isoformat(),
        'data_file': Path(data_file).name,
        'metrics': {
            'total_trades': metrics['total_trades'],
            'winning_trades': metrics['winning_trades'],
            'win_rate': metrics['win_rate'],
            'total_pnl': metrics['total_pnl'],
            'final_capital': metrics['final_capital'],
            'max_drawdown': metrics['max_drawdown'],
            'sharpe_ratio': metrics['sharpe_ratio']
        }
    }
    
    # Stampa risultati
    print("\nRISULTATI SIMULAZIONE")
    print("="*50)
    print(f"Trade totali: {metrics['total_trades']}")
    print(f"Trade vincenti: {metrics['winning_trades']}")
    print(f"Win rate: {metrics['win_rate']*100:.1f}%")
    print(f"P&L: ${metrics['total_pnl']:.2f}")
    print(f"Capitale finale: ${metrics['final_capital']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']*100:.1f}%")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Salva risultati se richiesto
    if output_file:
        print(f"\nSalvataggio risultati in {output_file}")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
