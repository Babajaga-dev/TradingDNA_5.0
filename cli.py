import argparse
from datetime import datetime
import traceback
import logging
from evolve import run_evolution
from simulate import run_simulation
from src.utils.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_config():
    """Stampa la configurazione genetica corrente"""
    print("\nCONFIGURAZIONE ATTUALE:")
    print("="*50)
    print("\nParametri genetici:")
    print(f"Population Size: {config.get('genetic.population_size')}")
    print(f"Generations: {config.get('genetic.generations')}")
    print(f"Mutation Rate: {config.get('genetic.mutation_rate')}")
    print(f"Elite Size: {config.get('genetic.elite_size')}")
    print(f"Tournament Size: {config.get('genetic.tournament_size')}")
    print(f"Min Trades: {config.get('genetic.min_trades')}")
    print(f"Parallel Processes: {config.get('genetic.parallel_processes')}")
    print(f"Batch Size: {config.get('genetic.batch_size')}")
    print("\n")

def main():
    """Funzione principale per il CLI"""
    parser = argparse.ArgumentParser(description='Trading System CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Comando evolve
    evolve_parser = subparsers.add_parser('evolve', help='Esegue l\'ottimizzazione genetica')
    evolve_parser.add_argument('--data', required=True, help='Percorso del file dati di mercato')
    evolve_parser.add_argument('--generations', type=int, 
                              default=config.get('genetic.generations'),
                              help=f'Numero di generazioni (default: {config.get("genetic.generations")})')
    evolve_parser.add_argument('--save-path', default='evolution_state.pkl',
                              help='Percorso dove salvare lo stato dell\'evoluzione')
    evolve_parser.add_argument('--population', type=int, 
                              default=config.get('genetic.population_size'),
                              help=f'Dimensione popolazione (default: {config.get("genetic.population_size")})')

    # Comando simulate
    simulate_parser = subparsers.add_parser('simulate', help='Esegue la simulazione di trading')
    simulate_parser.add_argument('--data', required=True, help='Percorso del file dati di mercato')
    simulate_parser.add_argument('--state-path', default='evolution_state.pkl',
                               help='Percorso del file di stato dell\'evoluzione')
    simulate_parser.add_argument('--output', help='Percorso del file di output dei risultati')

    args = parser.parse_args()

    try:
        start_time = datetime.now()
        
        if args.command == 'evolve':
            logger.info("Starting evolution process...")
            print("\nConfigurazione caricata da file config.yaml")
            print_config()
            
            # Mostra se ci sono modifiche dai parametri della CLI
            if args.generations != config.get('genetic.generations') or \
               args.population != config.get('genetic.population_size'):
                print("\nParametri modificati da linea di comando:")
                if args.generations != config.get('genetic.generations'):
                    print(f"Generations: {config.get('genetic.generations')} -> {args.generations}")
                if args.population != config.get('genetic.population_size'):
                    print(f"Population: {config.get('genetic.population_size')} -> {args.population}")
                print("")
            
            print(f"Starting evolution with {args.generations} generations...")
            try:
                run_evolution(
                    data_file=args.data,
                    generations=args.generations,
                    save_path=args.save_path,
                    population_size=args.population
                )
            except Exception as e:
                logger.error(f"Error during evolution: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
        elif args.command == 'simulate':
            print("Starting simulation...")
            run_simulation(
                data_file=args.data,
                evolution_state_path=args.state_path,
                output_file=args.output
            )
        
        elapsed = datetime.now() - start_time
        print(f"\nCompleted in {elapsed.total_seconds():.2f} seconds")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()