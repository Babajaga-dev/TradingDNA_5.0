import os
from pathlib import Path
from src.utils.data_loader import load_and_prepare_data
from src.optimization.genetic import run_genetic_trading_system
from src.models.simulator import TimeFrame
import matplotlib.pyplot as plt

def plot_optimization_results(optimizer):
    """Visualizza i risultati dell'ottimizzazione"""
    stats = optimizer.generation_stats
    generations = [s['generation'] for s in stats]
    best_fitness = [s['best_fitness'] for s in stats]
    avg_fitness = [s['avg_fitness'] for s in stats]
    
    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_fitness, label='Best Fitness')
    plt.plot(generations, avg_fitness, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Genetic Optimization Progress')
    plt.legend()
    plt.show()

def ensure_data_directory():
    """Ensure data directory exists and return path to data file"""
    # Get the project root directory (where main.py is located)
    project_root = Path(__file__).parent
    
    # Create data directory if it doesn't exist
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    return data_dir / 'market_data.csv'

def main():
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
    print(f"Caricamento dati da {data_file}...")
    try:
        data_dict = load_and_prepare_data(str(data_file))
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Parametri dell'ottimizzazione
    params = {
        'population_size': 100,
        'generations': 50,
        'initial_capital': 10000
    }
    
    # Esegui l'ottimizzazione
    print("\nAvvio ottimizzazione genetica...")
    print(f"Popolazione: {params['population_size']}")
    print(f"Generazioni: {params['generations']}")
    print(f"Capitale iniziale: ${params['initial_capital']}")
    
    best_gene, optimizer = run_genetic_trading_system(
        market_data=data_dict['1m'],
        timeframe=TimeFrame.M1,
        **params
    )
    
    # Stampa risultati
    print("\nMiglior Gene Trovato:")
    print(f"DNA: {best_gene.dna}")
    print("\nPerformance:")
    for metric, value in best_gene.performance_history.items():
        print(f"{metric}: {value}")
    
    # Visualizza risultati
    plot_optimization_results(optimizer)

if __name__ == "__main__":
    main()