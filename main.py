import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from src.optimization.ensemble_gene_optimizer import run_ensemble_optimization
from src.utils.data_loader import load_and_prepare_data
from src.models.simulator import TimeFrame, TradingSimulator
from src.utils.config import config
from src.models.gene import TorchGene, VolatilityAdaptiveGene, MomentumGene, PatternRecognitionGene

def plot_optimization_results(stats: Dict):
    """Visualizza i risultati dell'ottimizzazione"""
    generations = [s['generation'] for s in stats['generation_stats']]
    best_fitness = [s['best_fitness'] for s in stats['generation_stats']]
    avg_fitness = [s['avg_fitness'] for s in stats['generation_stats']]
    
    # Crea il plot principale
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot del fitness
    ax1.plot(generations, best_fitness, 'b-', label='Best Fitness')
    ax1.plot(generations, avg_fitness, 'r--', label='Average Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Score')
    ax1.set_title('Genetic Optimization Progress')
    ax1.legend()
    ax1.grid(True)
    
    # Plot del tempo per generazione
    generation_times = [s['time'] for s in stats['generation_stats']]
    ax2.bar(generations, generation_times, alpha=0.6)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Time per Generation')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_equity_curve(simulator: TradingSimulator):
    """Visualizza la curva di equity"""
    equity_data = pd.DataFrame(simulator.equity_curve, 
                             columns=['timestamp', 'equity'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_data['timestamp'], equity_data['equity'])
    plt.title('Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Capital ($)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def print_ensemble_details(ensemble: List[TorchGene]):
    """Stampa i dettagli dell'ensemble ottimizzato"""
    print("\nDETTAGLI ENSEMBLE OTTIMIZZATO")
    print("="*50)
    
    for i, gene in enumerate(ensemble):
        print(f"\n{i+1}. {type(gene).__name__}")
        print("-"*30)
        
        if isinstance(gene, TorchGene):
            print("Gene Base:")
            print(f"  Entry Indicator 1: {gene.dna['entry_indicator1']} {gene.dna['entry_indicator1_params']}")
            print(f"  Entry Indicator 2: {gene.dna['entry_indicator2']} {gene.dna['entry_indicator2_params']}")
            print(f"  Entry Operator: {gene.dna['entry_operator']}")
            print(f"  Exit Indicator 1: {gene.dna['exit_indicator1']} {gene.dna['exit_indicator1_params']}")
            print(f"  Exit Indicator 2: {gene.dna['exit_indicator2']} {gene.dna['exit_indicator2_params']}")
            print(f"  Exit Operator: {gene.dna['exit_operator']}")
            
        if isinstance(gene, VolatilityAdaptiveGene):
            print("Gene Volatilità:")
            print(f"  Timeperiod: {gene.dna['volatility_timeperiod']}")
            print(f"  Multiplier: {gene.dna['volatility_multiplier']:.2f}")
            print(f"  Base Position Size: {gene.dna['base_position_size']:.2f}%")
            
        if isinstance(gene, MomentumGene):
            print("Gene Momentum:")
            print(f"  Momentum Threshold: {gene.dna['momentum_threshold']}")
            print(f"  Trend Strength: {gene.dna['trend_strength_threshold']}")
            print(f"  Overbought Level: {gene.dna['overbought_level']}")
            print(f"  Oversold Level: {gene.dna['oversold_level']}")
            
        if isinstance(gene, PatternRecognitionGene):
            print("Gene Pattern:")
            print(f"  Required Patterns: {gene.dna['required_patterns']}")
            print(f"  Pattern Window: {gene.dna['pattern_window']}")
            print(f"  Confirmation Periods: {gene.dna['confirmation_periods']}")
        
        print(f"\nRisk Management:")
        print(f"  Position Size: {gene.dna['position_size_pct']:.2f}%")
        print(f"  Stop Loss: {gene.dna['stop_loss_pct']:.2f}%")
        print(f"  Take Profit: {gene.dna['take_profit_pct']:.2f}%")

def test_ensemble(ensemble: List[TorchGene], market_data: pd.DataFrame):
    """Testa l'ensemble ottimizzato su un set di dati"""
    simulator = TradingSimulator()
    simulator.add_market_data(TimeFrame.M1, market_data)
    
    print("\nAVVIO TEST ENSEMBLE")
    print("="*50)
    
    # Test ogni gene individualmente
    for i, gene in enumerate(ensemble):
        print(f"\nTest Gene {i+1}: {type(gene).__name__}")
        simulator.run_simulation(gene)
        metrics = simulator.get_performance_metrics()
        
        print(f"  Trade totali: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"  P&L: ${metrics['total_pnl']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Plot equity curve
    plot_equity_curve(simulator)

def setup_torch():
    """Configura PyTorch per prestazioni ottimali"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
    else:
        device = torch.device('cpu')
        torch.set_num_threads(torch.get_num_threads())
        print(f"\nUsing CPU with {torch.get_num_threads()} threads")
    return device

def ensure_data_directory():
    """Assicura che la directory dei dati esista e restituisce il path del file dati"""
    project_root = Path(__file__).parent
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    data_file = config.get("simulator.data_file", "market_data_BTC.csv")
    return data_dir / data_file

def main():
    # Setup PyTorch
    device = setup_torch()
    
    # Get data file path
    data_file = ensure_data_directory()
    
    # Check if data file exists
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        print("Please place your market data CSV file in the following location:")
        print(f"{data_file}")
        print("\nThe CSV file should contain the following columns:")
        print("timestamp, open, high, low, close, volume")
        return
    
    # Carica i dati
    print(f"\nCaricamento dati da {data_file}...")
    try:
        data_dict = load_and_prepare_data(str(data_file))
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Convert data to PyTorch tensors
    for timeframe in data_dict:
        prices = data_dict[timeframe]['close'].values
        data_dict[timeframe]['close_tensor'] = torch.tensor(prices, dtype=torch.float32).to(device)
    
    # Verifica la validità dei dati
    if not isinstance(data_dict['1m'], pd.DataFrame):
        print("Error: data_dict['1m'] is not a DataFrame")
        return
    
    # Stampa configurazione
    print("\nPARAMETRI DI CONFIGURAZIONE:")
    print(f"Popolazione: {config.get('genetic.population_size')}")
    print(f"Generazioni: {config.get('genetic.generations')}")
    print(f"Capitale iniziale: ${config.get('simulator.initial_capital')}")
    print(f"Geni abilitati:")
    print(f"Geni abilitati:")
    print(f"  - Base Gene: ✓")
    print(f"  - Volatility Gene: {'✓' if config.get('trading.volatility_gene.enabled') else '✗'}")
    print(f"  - Momentum Gene: {'✓' if config.get('trading.momentum_gene.enabled') else '✗'}")
    print(f"  - Pattern Gene: {'✓' if config.get('trading.pattern_gene.enabled') else '✗'}")
    
    # Esegui ottimizzazione ensemble
    print("\nAvvio ottimizzazione genetica ensemble...")
    best_ensemble, stats = run_ensemble_optimization(data_dict['1m'], device=device)
    
    # Stampa risultati dettagliati
    print_ensemble_details(best_ensemble)
    
    # Plot risultati
    plot_optimization_results(stats)
    
    # Test ensemble ottimizzato
    test_ensemble(best_ensemble, data_dict['1m'])
    
    # Clean up CUDA memory if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nFinal GPU Memory: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")

if __name__ == "__main__":
    main()