
import pandas as pd
import numpy as np
from simulator.simulator_processor import SimulationProcessor
from genetic.genetic_optimizer import GeneticOptimizer
from genes.rsi import RSIGene
from genes.macd import MACDGene
from genes.stochastic import StochasticOscillatorGene
from genes.atr import ATRGene
from genes.cci import CCIGene
from genes.volatility import VolatilityGene

class TradingPipeline:
    def __init__(self, config):
        self.config = config
        self.simulator = SimulationProcessor(config)
        self.optimizer = GeneticOptimizer(config)
        self.genes = {
            "RSI": RSIGene(config),
            "MACD": MACDGene(config),
            "Stochastic": StochasticOscillatorGene(config),
            "ATR": ATRGene(config),
            "CCI": CCIGene(config),
            "Volatility": VolatilityGene(config)
        }

    def load_data(self):
        """Load market data from the file specified in the configuration."""
        file_path = self.config['data']['market_data_file']
        data = pd.read_csv(file_path, parse_dates=['timestamp'])
        print(f"Loaded {len(data)} rows of market data.")
        return data

    def calculate_indicators(self, data):
        """Calculate all indicators and add them to the data frame."""
        print("Calculating indicators...")
        for name, gene in self.genes.items():
            if name == "MACD":
                macd, signal = gene.apply(data)
                data[f"{name}_MACD"] = macd
                data[f"{name}_Signal"] = signal
            elif name == "Stochastic":
                k, d = gene.apply(data)
                data[f"{name}_%K"] = k
                data[f"{name}_%D"] = d
            else:
                data[name] = gene.apply(data)
        print("Indicators calculated.")
        return data

    def generate_signals(self, data):
        """Generate trading signals based on indicator values and strategy parameters from the YAML config."""
        print("Generating trading signals...")
        data['signal'] = 0
        strategy_params = self.config['strategies']

        if "RSI" in strategy_params:
            rsi_params = strategy_params["RSI"]
            data.loc[data['RSI'] < rsi_params['buy_threshold'], 'signal'] += 1
            data.loc[data['RSI'] > rsi_params['sell_threshold'], 'signal'] -= 1

        if "MACD" in strategy_params:
            macd_params = strategy_params["MACD"]
            data.loc[(data['MACD_MACD'] > data['MACD_Signal']) & (data['MACD_MACD'] < macd_params['zero_cross_threshold']), 'signal'] += 1
            data.loc[(data['MACD_MACD'] < data['MACD_Signal']) & (data['MACD_MACD'] > macd_params['zero_cross_threshold']), 'signal'] -= 1

        if "Stochastic" in strategy_params:
            stoch_params = strategy_params["Stochastic"]
            data.loc[(data['Stochastic_%K'] < stoch_params['oversold']) & (data['Stochastic_%K'] > data['Stochastic_%D']), 'signal'] += 1
            data.loc[(data['Stochastic_%K'] > stoch_params['overbought']) & (data['Stochastic_%K'] < data['Stochastic_%D']), 'signal'] -= 1

        print("Signals generated.")
        return data

    def run_simulation(self, data):
        """Run the simulation with the generated signals."""
        print("Running simulation...")
        pnl, final_equity = self.simulator.run_simulation(data, data['signal'])
        print(f"Simulation complete. Final equity: {final_equity}")
        return pnl, final_equity

    def optimize(self, data):
        """Run genetic optimization to evolve trading strategies."""
        print("Starting genetic optimization...")
        best_strategy = self.optimizer.evolve(self.simulator, data, list(self.genes.values()))
        print("Optimization complete.")
        return best_strategy

    def save_report(self, pnl, final_equity):
        """Save simulation results to CSV files."""
        report_path = self.config['tests']['report_output']
        
        # Salva il PnL con timestamp
        pnl_df = pd.DataFrame({
            "Timestamp": range(len(pnl)),  # Indice temporale
            "PnL": pnl
        })
        pnl_df.to_csv(report_path.replace('.csv', '_pnl.csv'), index=False)
        
        # Salva il risultato finale in un file separato
        summary_df = pd.DataFrame({
            "Metric": ["Final Equity"],
            "Value": [final_equity]
        })
        summary_df.to_csv(report_path.replace('.csv', '_summary.csv'), index=False)
        
        print(f"Report salvato in:\n- {report_path.replace('.csv', '_pnl.csv')}\n- {report_path.replace('.csv', '_summary.csv')}")

    def run_pipeline(self):
        """Run the entire trading pipeline."""
        data = self.load_data()
        data = self.calculate_indicators(data)
        data = self.generate_signals(data)
        pnl, final_equity = self.run_simulation(data)
        self.save_report(pnl, final_equity)
        best_strategy = self.optimize(data)
        return pnl, final_equity, best_strategy

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    pipeline = TradingPipeline(config)
    results = pipeline.run_pipeline()
    print("Pipeline execution completed.", results)
