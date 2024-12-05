import argparse
from datetime import datetime
from evolve import run_evolution
from simulate import run_simulation

def main():
    parser = argparse.ArgumentParser(description='Trading System CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Evolve command
    evolve_parser = subparsers.add_parser('evolve')
    evolve_parser.add_argument('--data', required=True, help='Market data file path')
    evolve_parser.add_argument('--generations', type=int, default=50)
    evolve_parser.add_argument('--save-path', default='evolution_state.pkl')
    evolve_parser.add_argument('--population', type=int, default=100)

    # Simulate command
    simulate_parser = subparsers.add_parser('simulate')
    simulate_parser.add_argument('--data', required=True, help='Market data file path')
    simulate_parser.add_argument('--state-path', default='evolution_state.pkl')
    simulate_parser.add_argument('--output', help='Results output file')

    args = parser.parse_args()

    try:
        start_time = datetime.now()
        
        if args.command == 'evolve':
            print(f"Starting evolution with {args.generations} generations...")
            run_evolution(
                data_file=args.data,
                generations=args.generations,
                save_path=args.save_path,
                population_size=args.population
            )
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
        print(f"\nError: {str(e)}")
        raise

if __name__ == '__main__':
    main()