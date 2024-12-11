# src/optimization/ensemble.py
from typing import List, Dict, Tuple, Optional, Any, TypeVar, cast
import torch
import numpy as np
from multiprocessing import Pool, cpu_count
import pandas as pd
from datetime import datetime
import logging
from dataclasses import dataclass

from ..models.genes import (
    TorchGene, 
    VolatilityAdaptiveGene,
    MomentumGene, 
    PatternRecognitionGene,
    create_ensemble_gene
)
from ..models.common import Signal, SignalType, TimeFrame
from ..models.simulator import TradingSimulator
from ..utils.config import config

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=TorchGene)

@dataclass
class EnsembleMetrics:
    """Metriche per l'ensemble"""
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float

class EnsembleSignalCombiner:
    """Combina i segnali dai diversi geni dell'ensemble"""
    
    def __init__(self, ensemble_weights: Dict[str, float]):
        """
        Args:
            ensemble_weights: Dizionario con i pesi per ogni tipo di gene
        """
        self.weights = ensemble_weights

    def combine_signals(self, gene_signals: List[List[Signal]]) -> List[Signal]:
        """
        Combina i segnali usando un sistema di voto pesato
        
        Args:
            gene_signals: Lista di liste di segnali da ogni gene
            
        Returns:
            Lista di segnali combinati
        """
        if not gene_signals:
            return []
        
        try:
            combined_signals = []
            timestamp = gene_signals[0][0].timestamp if gene_signals[0] else None
            
            votes = {
                SignalType.LONG: 0.0,
                SignalType.SHORT: 0.0,
                SignalType.EXIT: 0.0
            }
            
            total_weight = 0.0
            for signals, weight in zip(gene_signals, self.weights.values()):
                if signals:
                    signal_type = signals[0].type
                    votes[signal_type] += weight
                    total_weight += weight
            
            if total_weight == 0:
                return []
                
            for signal_type in votes:
                votes[signal_type] /= total_weight
                
            winning_type = max(votes.items(), key=lambda x: x[1])
            if winning_type[1] >= 0.5:
                stop_losses: List[Tuple[float, float]] = []
                take_profits: List[Tuple[float, float]] = []
                
                for signals, weight in zip(gene_signals, self.weights.values()):
                    if signals and signals[0].type == winning_type[0]:
                        if signals[0].stop_loss is not None:
                            stop_losses.append((signals[0].stop_loss, weight))
                        if signals[0].take_profit is not None:
                            take_profits.append((signals[0].take_profit, weight))
                
                combined_signal = Signal(
                    type=winning_type[0],
                    timestamp=timestamp,
                    price=gene_signals[0][0].price,
                    metadata={"ensemble": True}
                )
                
                if stop_losses:
                    combined_signal.stop_loss = sum(sl * w for sl, w in stop_losses) / sum(w for _, w in stop_losses)
                if take_profits:
                    combined_signal.take_profit = sum(tp * w for tp, w in take_profits) / sum(w for _, w in take_profits)
                
                combined_signals.append(combined_signal)
            
            return combined_signals
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return []

class EnsembleGeneOptimizer:
    """Ottimizzatore per ensemble di geni"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: Device PyTorch opzionale
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parametri base
        self.population_size = config.get("genetic.population_size", 100)
        self.generations = config.get("genetic.generations", 50)
        self.mutation_rate = config.get("genetic.mutation_rate", 0.1)
        self.elite_size = config.get("genetic.elite_size", 10)
        self.tournament_size = config.get("genetic.tournament_size", 5)
        self.batch_size = config.get("genetic.batch_size", 32)
        self.min_trades = config.get("genetic.min_trades", 10)
        
        # Pesi ensemble
        self.ensemble_weights = {
            "base_gene": config.get("genetic.ensemble_weights.base_gene", 0.4),
            "volatility_gene": config.get("genetic.ensemble_weights.volatility_gene", 0.2),
            "momentum_gene": config.get("genetic.ensemble_weights.momentum_gene", 0.2),
            "pattern_gene": config.get("genetic.ensemble_weights.pattern_gene", 0.2)
        }
        
        # Setup
        self.population: List[List[TorchGene]] = []
        self.best_ensemble: Optional[List[TorchGene]] = None
        self.generation_stats: List[Dict[str, Any]] = []
        self.signal_combiner = EnsembleSignalCombiner(self.ensemble_weights)

    def evaluate_ensemble(self, ensemble: List[TorchGene], simulator: TradingSimulator) -> float:
        """
        Valuta un ensemble completo
        
        Args:
            ensemble: Lista di geni dell'ensemble
            simulator: Simulatore di trading
            
        Returns:
            Score di fitness
        """
        try:
            # Ottieni segnali da tutti i geni
            gene_signals: List[List[Signal]] = []
            for gene in ensemble:
                signals = gene.generate_signals(simulator.market_state)
                gene_signals.append(signals)
            
            # Combina segnali
            combined_signals = self.signal_combiner.combine_signals(gene_signals)
            
            # Simula trading con segnali combinati
            simulator._reset_simulation()
            for signal in combined_signals:
                simulator._process_signal(signal)
            
            metrics = simulator.get_performance_metrics()
            
            if metrics["total_trades"] < self.min_trades:
                return 0.0
            
            profit_weights = config.get("genetic.fitness_weights.profit_score", {})
            quality_weights = config.get("genetic.fitness_weights.quality_score", {})
            final_weights = config.get("genetic.fitness_weights.final_weights", {})
            
            profit_score = (
                min(1.0, metrics["total_pnl"] / simulator.initial_capital) * profit_weights.get("total_pnl", 0.4) +
                (1.0 - metrics["max_drawdown"]) * profit_weights.get("max_drawdown", 0.3) +
                max(0.0, metrics["sharpe_ratio"]) * profit_weights.get("sharpe_ratio", 0.3)
            )
            
            quality_score = (
                metrics["win_rate"] * quality_weights.get("win_rate", 0.6) +
                min(1.0, metrics["total_trades"] / 100.0) * quality_weights.get("trade_frequency", 0.4)
            )
            
            return (profit_score * final_weights.get("profit", 0.6) + 
                    quality_score * final_weights.get("quality", 0.4))
                    
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            return 0.0

    def create_initial_population(self) -> List[List[TorchGene]]:
        """
        Crea popolazione iniziale di ensemble
        
        Returns:
            Lista di ensemble
        """
        try:
            return [create_ensemble_gene(random_init=True) 
                    for _ in range(self.population_size)]
        except Exception as e:
            logger.error(f"Error creating initial population: {e}")
            return []

    def create_next_generation(self, evaluated_population: List[Tuple[List[TorchGene], float]]) -> None:
        """
        Crea la prossima generazione di ensemble
        
        Args:
            evaluated_population: Lista di tuple (ensemble, fitness)
        """
        try:
            sorted_population = sorted(evaluated_population, key=lambda x: x[1], reverse=True)
            new_population = [ensemble.copy() for ensemble, _ in sorted_population[:self.elite_size]]
            
            while len(new_population) < self.population_size:
                if np.random.random() < 0.8:  # 80% crossover
                    parent1 = self.select_parent(sorted_population)
                    parent2 = self.select_parent(sorted_population)
                    child = self.crossover_ensembles(parent1, parent2)
                    self.mutate_ensemble(child)
                else:
                    child = create_ensemble_gene(random_init=True)
                new_population.append(child)
            
            self.population = new_population
            
        except Exception as e:
            logger.error(f"Error creating next generation: {e}")

    def mutate_ensemble(self, ensemble: List[TorchGene]) -> None:
        """
        Muta ogni gene dell'ensemble
        
        Args:
            ensemble: Ensemble da mutare
        """
        try:
            for gene in ensemble:
                if np.random.random() < self.mutation_rate:
                    gene.mutate(np.random.uniform(0.2, 0.8))
        except Exception as e:
            logger.error(f"Error mutating ensemble: {e}")

    def crossover_ensembles(self, parent1: List[TorchGene], parent2: List[TorchGene]) -> List[TorchGene]:
        """
        Crossover tra due ensemble
        
        Args:
            parent1: Primo ensemble genitore
            parent2: Secondo ensemble genitore
            
        Returns:
            Nuovo ensemble figlio
        """
        try:
            child_ensemble: List[TorchGene] = []
            
            for gene1, gene2 in zip(parent1, parent2):
                child_gene = gene1.__class__(random_init=False)
                child_dna: Dict[str, Any] = {}
                
                for key in set(gene1.dna.keys()) | set(gene2.dna.keys()):
                    if np.random.random() < 0.5:
                        child_dna[key] = gene1.dna.get(key, gene2.dna[key])
                    else:
                        child_dna[key] = gene2.dna.get(key, gene1.dna[key])
                    
                    if isinstance(child_dna[key], (int, float)):
                        variation = np.random.uniform(-0.1, 0.1)
                        child_dna[key] *= (1.0 + variation)
                
                child_gene.dna = child_dna
                child_ensemble.append(child_gene)
            
            return child_ensemble
            
        except Exception as e:
            logger.error(f"Error in crossover: {e}")
            return create_ensemble_gene(random_init=True)

    def select_parent(self, evaluated_population: List[Tuple[List[TorchGene], float]]) -> List[TorchGene]:
        """
        Seleziona genitore usando tournament selection
        
        Args:
            evaluated_population: Lista di tuple (ensemble, fitness)
            
        Returns:
            Ensemble selezionato
        """
        try:
            tournament = np.random.choice(len(evaluated_population), size=self.tournament_size, replace=False)
            tournament_pop = [evaluated_population[i] for i in tournament]
            
            weights = [max(0.1, score) for _, score in tournament_pop]
            weights = np.array(weights) / sum(weights)
            
            selected_idx = np.random.choice(len(tournament_pop), p=weights)
            return tournament_pop[selected_idx][0]
            
        except Exception as e:
            logger.error(f"Error selecting parent: {e}")
            return create_ensemble_gene(random_init=True)

    def evaluate_ensemble_parallel(self, args: Tuple[int, List[TorchGene], Dict[str, Any]]) -> Tuple[List[TorchGene], float]:
        """
        Valutazione parallela di un ensemble
        
        Args:
            args: Tupla (indice, ensemble, dati di mercato)
            
        Returns:
            Tupla (ensemble, fitness)
        """
        try:
            ensemble_idx, ensemble, market_data_dict = args
            
            simulator = TradingSimulator()
            
            for timeframe_str, data in market_data_dict.items():
                timeframe = TimeFrame(timeframe_str)
                df = pd.DataFrame(data)
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                simulator.add_market_data(timeframe, df)
                
            return ensemble, self.evaluate_ensemble(ensemble, simulator)
            
        except Exception as e:
            logger.error(f"Error in parallel evaluation: {e}")
            return ensemble, 0.0

    def optimize(self, simulator: TradingSimulator) -> Tuple[List[TorchGene], Dict[str, Any]]:
        """
        Esegue ottimizzazione dell'ensemble
        
        Args:
            simulator: Simulatore di trading
            
        Returns:
            Tupla (miglior ensemble, statistiche)
        """
        try:
            self.population = self.create_initial_population()
            best_fitness = float('-inf')
            generations_without_improvement = 0
            
            # Prepara dati per parallelizzazione
            market_data_dict = {}
            for timeframe, data in simulator.market_data.items():
                df_dict = pd.DataFrame([{
                    'timestamp': d.timestamp,
                    'open': d.open,
                    'high': d.high,
                    'low': d.low,
                    'close': d.close,
                    'volume': d.volume
                } for d in data])
                market_data_dict[timeframe.value] = df_dict.to_dict('records')

            num_workers = min(cpu_count(), 8)  # Limita a max 8 workers
            
            for generation in range(self.generations):
                generation_start = datetime.now()
                
                # Valutazione parallela
                eval_args = [(i, ensemble, market_data_dict) 
                            for i, ensemble in enumerate(self.population)]
                
                evaluated_population: List[Tuple[List[TorchGene], float]] = []
                total_batches = (len(eval_args) + self.batch_size - 1) // self.batch_size
                
                for batch_idx in range(total_batches):
                    batch_start = batch_idx * self.batch_size
                    batch_end = min(batch_start + self.batch_size, len(eval_args))
                    current_batch = eval_args[batch_start:batch_end]
                    
                    with Pool(num_workers) as pool:
                        batch_results = list(pool.imap_unordered(
                            self.evaluate_ensemble_parallel, 
                            current_batch
                        ))
                        evaluated_population.extend(batch_results)
                
                evaluated_population.sort(key=lambda x: x[1], reverse=True)
                current_best = evaluated_population[0]
                
                if current_best[1] > best_fitness:
                    best_fitness = current_best[1]
                    self.best_ensemble = current_best[0]
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Statistiche generazione
                generation_time = (datetime.now() - generation_start).total_seconds()
                avg_fitness = np.mean([f for _, f in evaluated_population])
                
                self.generation_stats.append({
                    'generation': generation + 1,
                    'best_fitness': current_best[1],
                    'avg_fitness': avg_fitness,
                    'time': generation_time
                })
                
                if generations_without_improvement >= 10:
                    break
                    
                self.create_next_generation(evaluated_population)
            
            if self.best_ensemble is None:
                self.best_ensemble = create_ensemble_gene(random_init=True)
                best_fitness = 0.0
            
            return self.best_ensemble, {
                'best_fitness': best_fitness,
                'generation_stats': self.generation_stats
            }
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return create_ensemble_gene(random_init=True), {
                'error': str(e),
                'best_fitness': 0.0,
                'generation_stats': []
            }
