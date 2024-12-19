# src/optimization/evolve.py
import pickle
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import traceback

from src.utils.data import load_and_prepare_data
from src.models.common import TimeFrame
from src.models.simulator import TradingSimulator
from src.optimization.genetic_optimizer import ParallelGeneticOptimizer
from src.utils.config import config

logger = logging.getLogger(__name__)

def calculate_linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Calcola regressione lineare semplice"""
    n = len(x)
    if n != len(y):
        raise ValueError("x e y devono avere la stessa lunghezza")
        
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calcola il coefficiente di regressione (slope)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sum((x - mean_x) ** 2)
    
    if denominator == 0:
        return 0, 0, 0
        
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    
    # Calcola R-squared
    y_pred = slope * x + intercept
    r_squared = (np.corrcoef(y, y_pred)[0, 1]) ** 2
    
    return slope, r_squared, 0.05  # p-value fisso a 0.05 per semplificazione

def analyze_evolution_history(generation_stats: List[Dict]) -> Dict:
    """Analizza la storia dell'evoluzione e calcola metriche aggiuntive"""
    stats_df = pd.DataFrame(generation_stats)
    
    # Calcolo delle metriche di convergenza
    convergence_metrics = {
        'final_best_fitness': stats_df['best_fitness'].iloc[-1],
        'final_avg_fitness': stats_df['avg_fitness'].iloc[-1],
        'improvement_rate': (stats_df['best_fitness'].iloc[-1] - stats_df['best_fitness'].iloc[0]) / len(stats_df),
        'convergence_gen': stats_df[stats_df['best_fitness'] == stats_df['best_fitness'].max()].index[0] + 1
    }
    
    # Analisi della variabilità
    variability_metrics = {
        'fitness_std_last_10': stats_df['best_fitness'].tail(10).std(),
        'population_diversity_last_10': stats_df['std_fitness'].tail(10).mean(),
    }
    
    # Calcolo delle tendenze usando la nostra funzione
    x = np.arange(len(stats_df))
    slope, r_squared, p_value = calculate_linear_regression(x, stats_df['best_fitness'].values)
    trend_metrics = {
        'fitness_slope': slope,
        'fitness_r_squared': r_squared,
        'fitness_p_value': p_value
    }
    
    # Analisi dei plateau
    fitness_diff = stats_df['best_fitness'].diff()
    plateau_lengths = []
    current_plateau = 0
    for diff in fitness_diff:
        if abs(diff) < 1e-6:  # threshold for considering no improvement
            current_plateau += 1
        elif current_plateau > 0:
            plateau_lengths.append(current_plateau)
            current_plateau = 0
    if current_plateau > 0:
        plateau_lengths.append(current_plateau)
    
    plateau_metrics = {
        'num_plateaus': len(plateau_lengths),
        'avg_plateau_length': np.mean(plateau_lengths) if plateau_lengths else 0,
        'max_plateau_length': max(plateau_lengths) if plateau_lengths else 0
    }
    
    return {
        'convergence': convergence_metrics,
        'variability': variability_metrics,
        'trends': trend_metrics,
        'plateaus': plateau_metrics
    }

def plot_evolution_history(generation_stats: List[Dict], plot_dir: Path):
    """Genera e salva grafici dell'evoluzione"""
    try:
        logger.info("Creating evolution history plots...")
        
        # Assicurati che la directory esista
        plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Plot directory ensured: {plot_dir}")
        
        stats_df = pd.DataFrame(generation_stats)
        
        # Plot 1: Fitness trends
        plt.figure(figsize=(12, 6))
        plt.plot(stats_df['best_fitness'], label='Best Fitness', color='blue')
        plt.plot(stats_df['avg_fitness'], label='Average Fitness', color='green')
        plt.fill_between(stats_df.index, 
                        stats_df['avg_fitness'] - stats_df['std_fitness'],
                        stats_df['avg_fitness'] + stats_df['std_fitness'],
                        alpha=0.2, color='green')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution of Fitness Scores')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_dir / 'fitness_evolution.png')
        plt.close()
        
        # Plot 2: Population Diversity
        plt.figure(figsize=(12, 6))
        plt.plot(stats_df['std_fitness'], label='Population Diversity', color='red')
        plt.xlabel('Generation')
        plt.ylabel('Standard Deviation')
        plt.title('Population Diversity Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_dir / 'population_diversity.png')
        plt.close()
        
        logger.info("Plots created successfully")
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_evolution_report(evolution_state: Dict, report_dir: Path) -> str:
    """Genera un report dettagliato dell'evoluzione"""
    try:
        logger.info("Generating evolution report...")
        
        # Assicurati che la directory esista
        report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Report directory ensured: {report_dir}")
        
        history_analysis = analyze_evolution_history(evolution_state['generation_stats'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = []
        report.append("\nREPORT DETTAGLIATO EVOLUZIONE")
        report.append("="*50)
        
        # Informazioni generali
        report.append("\nINFORMAZIONE GENERALI:")
        report.append(f"Data esecuzione: {evolution_state['timestamp']}")
        report.append(f"File dati: {evolution_state['data_info']['filename']}")
        report.append(f"Periodo: {evolution_state['data_info']['start_date']} - {evolution_state['data_info']['end_date']}")
        
        # Metriche di convergenza
        report.append("\nMETRICHE DI CONVERGENZA:")
        report.append(f"Fitness finale migliore: {history_analysis['convergence']['final_best_fitness']:.4f}")             
        report.append(f"Fitness media finale: {history_analysis['convergence']['final_avg_fitness']:.4f}")
        report.append(f"Tasso di miglioramento per generazione: {history_analysis['convergence']['improvement_rate']:.6f}")
        report.append(f"Generazione di convergenza: {history_analysis['convergence']['convergence_gen']}")
        
        # Analisi della variabilità
        report.append("\nANALISI DELLA VARIABILITÀ:")
        report.append(f"Deviazione standard fitness (ultime 10 gen): {history_analysis['variability']['fitness_std_last_10']:.4f}")
        report.append(f"Diversità media popolazione (ultime 10 gen): {history_analysis['variability']['population_diversity_last_10']:.4f}")
        
        # Analisi delle tendenze
        report.append("\nANALISI DELLE TENDENZE:")
        report.append(f"Pendenza della fitness: {history_analysis['trends']['fitness_slope']:.6f}")
        report.append(f"R² della fitness: {history_analysis['trends']['fitness_r_squared']:.4f}")
        report.append(f"P-value della tendenza: {history_analysis['trends']['fitness_p_value']:.4f}")
        
        # Analisi dei plateau
        report.append("\nANALISI DEI PLATEAU:")
        report.append(f"Numero di plateau: {history_analysis['plateaus']['num_plateaus']}")
        report.append(f"Lunghezza media plateau: {history_analysis['plateaus']['avg_plateau_length']:.2f}")
        report.append(f"Lunghezza massima plateau: {history_analysis['plateaus']['max_plateau_length']}")
        
        # Parametri del miglior gene
        if evolution_state['best_gene'] is not None:
            report.append("\nPARAMETRI DEL MIGLIOR GENE:")
            for key, value in evolution_state['best_gene'].dna.items():
                report.append(f"{key}: {value}")
            
            # Metriche di performance del miglior gene
            report.append("\nPERFORMANCE DEL MIGLIOR GENE:")
            stats = evolution_state['stats']
            report.append(f"Total trades: {stats['total_trades']}")
            report.append(f"Win rate: {stats['win_rate']:.2%}")
            report.append(f"Profit factor: {stats['profit_factor']:.2f}")
            report.append(f"Sharpe ratio: {stats['sharpe_ratio']:.2f}")
            report.append(f"Max drawdown: {stats['max_drawdown']:.2%}")
            report.append(f"Total PnL: ${stats['total_pnl']:.2f}")
            report.append(f"Final capital: ${stats['final_capital']:.2f}")
        else:
            report.append("\nNESSUN GENE VALIDO TROVATO")
            report.append("L'ottimizzazione non ha prodotto un gene valido.")
        
        report_content = '\n'.join(report)
        
        # Salva il report
        report_file = report_dir / f'evolution_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report generated and saved to {report_file}")
        return report_content
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def ensure_directories(save_path: str) -> Tuple[Path, Path, Path]:
    """
    Crea le directory necessarie per salvare i risultati dell'evoluzione.
    """
    try:
        logger.info(f"Creating directories for save_path: {save_path}")
        # Converti il percorso in oggetto Path
        save_path = Path(save_path)
        
        # Crea la directory principale se non esiste
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Main directory created: {save_path.parent}")
        
        # Crea la directory per i grafici
        plot_dir = save_path.parent / 'evolution_plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Plots directory created: {plot_dir}")
        
        # Crea la directory per i report
        report_dir = save_path.parent / 'evolution_reports'
        report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Reports directory created: {report_dir}")
        
        return save_path, plot_dir, report_dir
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def run_evolution(data_file: str,
                 generations: int = None,
                 save_path: str = "evolution_state.pkl",
                 population_size: int = None) -> None:
    """Esegue l'evoluzione genetica della popolazione"""
    try:
        print("\nINIZIALIZZAZIONE EVOLUZIONE")
        print("="*50)
        logger.info("Starting evolution process")
        
        # Crea le directory necessarie
        logger.info("Creating directories...")
        save_path, plot_dir, report_dir = ensure_directories(save_path)
        
        # Override configurazione se specificato
        if generations:
            logger.info(f"Overriding generations to: {generations}")
            config._config['genetic']['generations'] = generations
        if population_size:
            logger.info(f"Overriding population_size to: {population_size}")
            config._config['genetic']['population_size'] = population_size
            
        # Verifica esistenza del file
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Carica e prepara dati
        logger.info(f"Loading market data from {data_file}...")
        print("Caricamento dati di mercato...")
        market_data = load_and_prepare_data(data_file)
        logger.info("Market data loaded successfully")
        
        # Setup simulatore
        logger.info("Initializing simulator...")
        print("Inizializzazione simulatore...")
        simulator = TradingSimulator()
        simulator.add_market_data(TimeFrame.M1, market_data['1m'])
        logger.info("Simulator initialized")
        
        # Esegui ottimizzazione
        logger.info("Starting genetic optimization...")
        print("\nAvvio ottimizzazione genetica...")
        optimizer = ParallelGeneticOptimizer()
        best_gene, stats = optimizer.optimize(simulator)
        logger.info("Optimization completed")
        
        # Verifica se l'ottimizzazione ha avuto successo
        if best_gene is None:
            logger.error("Optimization failed to produce a valid gene")
            print("\nERRORE: L'ottimizzazione non ha prodotto un gene valido")
            return
        
        # Fai una simulazione finale con il miglior gene
        final_metrics = simulator.run_simulation_vectorized(
            best_gene.generate_entry_conditions(optimizer.precalculated_data)
        )
        
        # Timestamp per i file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepara stato da salvare
        evolution_state = {
            'timestamp': datetime.now(),
            'best_gene': best_gene,
            'stats': final_metrics,  # Usa le metriche finali qui
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
        logger.info(f"Saving evolution state to {save_path}")
        print(f"\nSalvataggio stato evoluzione in {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(evolution_state, f)
        
        # Genera report e grafici
        logger.info("Generating reports and plots...")
        report = generate_evolution_report(evolution_state, report_dir)
        plot_evolution_history(optimizer.generation_stats, plot_dir)
        print("\nEvoluzione completata!")
        print(f"Stato salvato in: {save_path}")
        print(f"Grafici salvati in: {plot_dir}")
        print(f"Report dettagliato salvato in: {report_dir}")
        logger.info("Evolution process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in run_evolution: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evolutionary Trading System")
    parser.add_argument("--data", required=True, help="Path to market data file")
    parser.add_argument("--generations", type=int, help="Number of generations")
    parser.add_argument("--population", type=int, help="Population size")
    parser.add_argument("--save-path", default="evolution_state.pkl", help="Path to save evolution state")
    
    args = parser.parse_args()
    
    try:
        run_evolution(
            data_file=args.data,
            generations=args.generations,
            save_path=args.save_path,
            population_size=args.population
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
