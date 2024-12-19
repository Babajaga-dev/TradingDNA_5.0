import argparse
from datetime import datetime
import traceback
import logging
from evolve import run_evolution
from simulate import run_simulation
from src.utils.config import config
import os

def setup_logging(debug: bool = False):
    """Configura il sistema di logging"""
    level = logging.DEBUG if debug else logging.INFO
    
    # Formatter condiviso per tutti i logger
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        '%Y-%m-%d %H:%M:%S'
    )
    
    # Configura il logger del CLI
    cli_logger = logging.getLogger(__name__)
    cli_logger.setLevel(level)
    cli_logger.propagate = False
    cli_logger.handlers = []
    cli_handler = logging.StreamHandler()
    cli_handler.setFormatter(formatter)
    cli_handler.setLevel(level)
    cli_logger.addHandler(cli_handler)

    # Configura i logger dei moduli principali
    loggers = [
        'src.models.simulator',
        'src.models.simulator_processor',
        'src.models.position_manager',
        'src.optimization.genetic_optimizer',
        'src.optimization.genetic_evaluation',
        'src.optimization.genetic_monitoring',
        'src.models.genes.base'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = False  # Disabilita propagazione al parent logger
        logger.handlers = []  # Rimuovi handler esistenti
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        if "the following arguments are required: --config" in message:
            self.print_help()
            print("\nErrore: File di configurazione non specificato!")
            print("\nEsempio di utilizzo:")
            print("  python cli.py --config config.yaml evolve")
            print("  python cli.py --config config.yaml simulate")
            exit(2)
        super().error(message)

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
    print(f"Batch Size: {config.get('genetic.batch_size')}")
    print("\n")

def get_default_data_path():
    """Costruisce il path dei dati di default dal config"""
    output_folder = config.get('download.output_folder', 'data')
    data_file = config.get('simulator.data_file')
    if not data_file:
        raise ValueError("data_file non specificato nella configurazione")
    return os.path.join(output_folder, f"{data_file}.csv")

def main():
    """Funzione principale per il CLI"""
    parser = CustomArgumentParser(description='Trading System CLI')
    
    # Argomenti globali
    parser.add_argument('--config', required=True,
                       help='Percorso del file di configurazione (obbligatorio)')
    parser.add_argument('--debug', action='store_true',
                       help='Attiva il logging in modalità debug')
    
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Comando evolve
    evolve_parser = subparsers.add_parser('evolve', help='Esegue l\'ottimizzazione genetica')
    evolve_parser.add_argument('--data', required=False, 
                              help='Percorso del file dati di mercato (opzionale, default dal config)')
    evolve_parser.add_argument('--generations', type=int, 
                              default=None,
                              help='Numero di generazioni (se non specificato, usa il valore dal config)')
    evolve_parser.add_argument('--save-path', default='evolution_state.pkl',
                              help='Percorso dove salvare lo stato dell\'evoluzione')
    evolve_parser.add_argument('--population', type=int, 
                              default=None,
                              help='Dimensione popolazione (se non specificato, usa il valore dal config)')

    # Comando simulate
    simulate_parser = subparsers.add_parser('simulate', help='Esegue la simulazione di trading')
    simulate_parser.add_argument('--data', required=False,
                               help='Percorso del file dati di mercato (opzionale, default dal config)')
    simulate_parser.add_argument('--state-path', default='evolution_state.pkl',
                               help='Percorso del file di stato dell\'evoluzione')
    simulate_parser.add_argument('--output', help='Percorso del file di output dei risultati')

    args = parser.parse_args()

    try:
        # Setup logging
        setup_logging(args.debug)
        
        # Carica la configurazione dal file specificato
        config.load_config(args.config)
        
        start_time = datetime.now()
        
        if args.command == 'evolve':
            logging.getLogger(__name__).info("Starting evolution process...")
            print(f"\nConfigurazione caricata da {args.config}")
            print_config()
            
            # Usa i valori dalla CLI se specificati, altrimenti usa quelli dal config
            generations = args.generations if args.generations is not None else config.get('genetic.generations')
            population_size = args.population if args.population is not None else config.get('genetic.population_size')
            data_path = args.data if args.data is not None else get_default_data_path()
            
            # Mostra se ci sono modifiche dai parametri della CLI
            if args.generations is not None or args.population is not None:
                print("\nParametri modificati da linea di comando:")
                if args.generations is not None:
                    print(f"Generations: {config.get('genetic.generations')} -> {generations}")
                if args.population is not None:
                    print(f"Population: {config.get('genetic.population_size')} -> {population_size}")
                print("")
            
            print(f"Starting evolution with {generations} generations...")
            print(f"Using data file: {data_path}")
            try:
                run_evolution(
                    data_file=data_path,
                    generations=generations,
                    save_path=args.save_path,
                    population_size=population_size
                )
            except Exception as e:
                logging.getLogger(__name__).error(f"Error during evolution: {str(e)}")
                logging.getLogger(__name__).error(traceback.format_exc())
                raise
            
        elif args.command == 'simulate':
            print("Starting simulation...")
            data_path = args.data if args.data is not None else get_default_data_path()
            print(f"Using data file: {data_path}")
            run_simulation(
                data_file=data_path,
                evolution_state_path=args.state_path,
                output_file=args.output
            )
        
        elapsed = datetime.now() - start_time
        print(f"\nCompleted in {elapsed.total_seconds():.2f} seconds")

    except Exception as e:
        logging.getLogger(__name__).error(f"Error in main: {str(e)}")
        logging.getLogger(__name__).error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
