import yaml
from simulator.simulator_processor import SimulationProcessor
from simulator_pipeline import TradingPipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    pipeline = TradingPipeline(config)
    results = pipeline.run_pipeline()
    print("Pipeline execution completed.", results)
