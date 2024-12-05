from typing import List, Dict
from datetime import datetime
import numpy as np
import pandas as pd
from src.models.common import Signal, SignalType, MarketData, TimeFrame
from src.utils.config import config
from src.models.simulator import TradingSimulator, Position
from src.models.gene import TradingGene, VolatilityAdaptiveGene, MomentumGene, PatternRecognitionGene
    

class EnsembleSimulator(TradingSimulator):  # Continua dalla parte 1
    
    def run_ensemble_simulation(self, ensemble: List[TradingGene]) -> Dict:
        """Esegue la simulazione per l'intero ensemble di geni"""
        min_candles = config.get("simulator.min_candles", 50)
        
        if TimeFrame.M1 not in self.market_data:
            raise ValueError("Dati al minuto (M1) non trovati")
        
        # Reset stato simulatore
        self._reset_simulation_state()
        
        # Identifica tipi di gene nell'ensemble
        gene_types = self._identify_gene_types(ensemble)
        
        total_candles = len(self.market_data[TimeFrame.M1])
        self.simulation_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'consensus_reached': 0,
            'gene_signals': {i: 0 for i in range(len(ensemble))}
        }
        
        print(f"\nAvvio simulazione ensemble con {len(ensemble)} geni")
        print(f"Composizione ensemble: {', '.join(gene_types)}")
        
        # Loop principale simulazione
        for i, current_data in enumerate(self.market_data[TimeFrame.M1]):
            try:
                # Aggiorna stato corrente
                self.current_time = current_data.timestamp
                self._print_progress(i, total_candles)
                
                # Gestisce posizioni esistenti
                self._update_positions(current_data)
                
                # Verifica dati sufficienti
                historical_data = self._get_historical_data(current_data)
                if len(historical_data) < min_candles:
                    continue
                
                # Genera e processa segnali
                self._process_ensemble_signals(ensemble, historical_data, gene_types, current_data)
                
            except Exception as e:
                self._handle_simulation_error(e, i)
        
        # Finalizza simulazione
        self._finalize_simulation()
        
        return self.get_simulation_results()

    def _reset_simulation_state(self):
        """Reset dello stato del simulatore"""
        self.capital = self.initial_capital
        self.positions.clear()
        self.historical_positions.clear()
        self.equity_curve = [(self.market_data[TimeFrame.M1][0].timestamp, self.capital)]
        self.gene_performance.clear()

    def _identify_gene_types(self, ensemble: List[TradingGene]) -> List[str]:
        """Identifica i tipi di gene nell'ensemble"""
        gene_types = []
        for gene in ensemble:
            if isinstance(gene, VolatilityAdaptiveGene):
                gene_types.append("volatility_gene")
            elif isinstance(gene, MomentumGene):
                gene_types.append("momentum_gene")
            elif isinstance(gene, PatternRecognitionGene):
                gene_types.append("pattern_gene")
            else:
                gene_types.append("base_gene")
        return gene_types

    def _print_progress(self, current_candle: int, total_candles: int):
        """Stampa il progresso della simulazione"""
        if current_candle % max(1, total_candles // 20) == 0:
            progress = (current_candle / total_candles) * 100
            current_profit = self.capital - self.initial_capital
            print(f"\rProgresso: {progress:.1f}% - P&L: ${current_profit:.2f} - "
                  f"Posizioni: {len(self.positions)} - "
                  f"Segnali: {self.simulation_stats['signals_generated']}", end="")

    def _get_historical_data(self, current_data: MarketData) -> List[MarketData]:
        """Ottiene i dati storici fino al momento corrente"""
        return [d for d in self.market_data[TimeFrame.M1]
                if d.timestamp <= current_data.timestamp]

    def _process_ensemble_signals(self, ensemble: List[TradingGene], 
                                historical_data: List[MarketData],
                                gene_types: List[str],
                                current_data: MarketData):
        """Processa i segnali dell'ensemble"""
        all_signals = []
        
        # Raccogli segnali da ogni gene
        for i, gene in enumerate(ensemble):
            signals = gene.generate_signals(historical_data)
            if signals:
                self.simulation_stats['gene_signals'][i] += 1
                self.simulation_stats['signals_generated'] += len(signals)
                all_signals.append(signals)
            else:
                all_signals.append([])
        
        # Combina e processa segnali
        if any(all_signals):
            combined_signals = self.combine_signals(all_signals, gene_types)
            if combined_signals:
                self.simulation_stats['consensus_reached'] += 1
                for signal in combined_signals:
                    self._process_signal(signal, current_data, None)
                    self.simulation_stats['trades_executed'] += 1

    def _handle_simulation_error(self, error: Exception, candle_index: int):
        """Gestisce gli errori durante la simulazione"""
        print(f"\nErrore durante la simulazione alla candela {candle_index}:")
        print(f"Tipo errore: {type(error).__name__}")
        print(f"Descrizione: {str(error)}")
        raise error

    def _finalize_simulation(self):
        """Finalizza la simulazione e prepara le statistiche"""
        print("\n\nSimulazione ensemble completata")
        self._print_simulation_summary()

    def _print_simulation_summary(self):
        """Stampa il riepilogo della simulazione"""
        print("\nRIEPILOGO SIMULAZIONE")
        print("="*50)
        print(f"Segnali totali generati: {self.simulation_stats['signals_generated']}")
        print(f"Trade eseguiti: {self.simulation_stats['trades_executed']}")
        print(f"Consensus raggiunti: {self.simulation_stats['consensus_reached']}")
        print("\nContribuzione per gene:")
        for gene_id, signals in self.simulation_stats['gene_signals'].items():
            print(f"  Gene {gene_id}: {signals} segnali")
        self._print_gene_statistics()

    def get_simulation_results(self) -> Dict:
        """Restituisce i risultati completi della simulazione"""
        metrics = self.get_performance_metrics()
        return {
            'performance_metrics': metrics,
            'simulation_stats': self.simulation_stats,
            'gene_metrics': self.get_gene_metrics(),
            'equity_curve': self.equity_curve,
            'gene_performance': self.gene_performance
        }

    def get_gene_metrics(self) -> Dict:
        """Restituisce le metriche dettagliate per ogni gene"""
        metrics = {}
        for gene_id, performances in self.gene_performance.items():
            if performances:
                metrics[gene_id] = {
                    'total_trades': len(performances),
                    'total_pnl': sum(performances),
                    'avg_pnl': np.mean(performances),
                    'win_rate': len([p for p in performances if p > 0]) / len(performances),
                    'std_dev': np.std(performances) if len(performances) > 1 else 0,
                    'sharpe': self._calculate_gene_sharpe(performances),
                    'max_drawdown': self._calculate_gene_drawdown(performances)
                }
        return metrics

    def _calculate_gene_sharpe(self, performances: List[float]) -> float:
        """Calcola il Sharpe Ratio per un gene"""
        if len(performances) < 2:
            return 0
        returns = np.diff(performances) / np.abs(performances[:-1])
        return np.sqrt(252) * (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0

    def _calculate_gene_drawdown(self, performances: List[float]) -> float:
        """Calcola il Maximum Drawdown per un gene"""
        cumulative = np.cumsum(performances)
        max_dd = 0
        peak = cumulative[0]
        for value in cumulative[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    def _print_gene_statistics(self):
        """Stampa statistiche dettagliate per ogni gene"""
        print("\nSTATISTICHE PER GENE")
        print("="*50)
        
        metrics = self.get_gene_metrics()
        for gene_id, metric in metrics.items():
            print(f"\nGene {gene_id}:")
            print(f"  Trade totali: {metric['total_trades']}")
            print(f"  P&L totale: ${metric['total_pnl']:.2f}")
            print(f"  P&L medio: ${metric['avg_pnl']:.2f}")
            print(f"  Win Rate: {metric['win_rate']*100:.1f}%")
            print(f"  Sharpe Ratio: {metric['sharpe']:.2f}")
            print(f"  Max Drawdown: {metric['max_drawdown']*100:.1f}%")
    