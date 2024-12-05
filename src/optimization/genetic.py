# src/optimization/genetic.py
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from multiprocessing import Pool
import pandas as pd

from ..models.genes import TradingGene
from ..models.simulator import TradingSimulator
from ..models.common import TimeFrame
from ..utils.config import config

class ParallelGeneticOptimizer:
    def __init__(self):
        # Carica configurazione
        self.population_size = config.get("genetic.population_size", 100)
        self.generations = config.get("genetic.generations", 50)
        self.mutation_rate = config.get("genetic.mutation_rate", 0.1)
        self.elite_size = config.get("genetic.elite_size", 10)
        self.tournament_size = config.get("genetic.tournament_size", 5)
        self.min_trades = config.get("genetic.min_trades", 10)
        self.batch_size = config.get("genetic.batch_size", 32)
        
        # Setup parallelizzazione
        configured_processes = config.get("genetic.parallel_processes", 0)
        self.num_processes = min(8, configured_processes) if configured_processes > 0 else 8
        
        # Pesi fitness
        self.weights = {
            "profit_score": {
                "total_pnl": config.get("genetic.fitness_weights.profit_score.total_pnl", 0.4),
                "max_drawdown": config.get("genetic.fitness_weights.profit_score.max_drawdown", 0.3),
                "sharpe_ratio": config.get("genetic.fitness_weights.profit_score.sharpe_ratio", 0.3)
            },
            "quality_score": {
                "win_rate": config.get("genetic.fitness_weights.quality_score.win_rate", 0.6),
                "trade_frequency": config.get("genetic.fitness_weights.quality_score.trade_frequency", 0.4)
            },
            "final": {
                "profit": config.get("genetic.fitness_weights.final_weights.profit", 0.6),
                "quality": config.get("genetic.fitness_weights.final_weights.quality", 0.4)
            }
        }
        
        self.population = []
        self.generation_stats = []

    def evaluate_gene_parallel(self, args: Tuple) -> Tuple[TradingGene, float]:
        """Valuta un singolo gene nel pool parallelo"""
        gene_idx, gene, market_data_dict = args
        
        # Setup simulatore
        simulator = TradingSimulator()
        for timeframe_str, data in market_data_dict.items():
            timeframe = TimeFrame(timeframe_str)
            df = pd.DataFrame(data)
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            simulator.add_market_data(timeframe, df)
        
        # Esegui simulazione
        simulator.run_simulation(gene)
        metrics = simulator.get_performance_metrics()
        
        # Calcola fitness
        fitness = self.calculate_fitness(metrics, simulator.initial_capital)
        gene.fitness_score = fitness
        gene.performance_history = metrics
        
        return gene, fitness

    def evaluate_population(self, simulator: TradingSimulator) -> List[Tuple[TradingGene, float]]:
        """Valuta l'intera popolazione in parallelo"""
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
            
        # Prepara argomenti per valutazione parallela
        eval_args = [(i, gene, market_data_dict) for i, gene in enumerate(self.population)]
        
        # Valuta in batch paralleli
        results = []
        total_batches = (len(eval_args) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(eval_args))
            current_batch = eval_args[batch_start:batch_end]
            
            with Pool(processes=self.num_processes) as pool:
                batch_results = list(pool.imap_unordered(self.evaluate_gene_parallel, current_batch))
                results.extend(batch_results)
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def calculate_fitness(self, metrics: Dict, initial_capital: float) -> float:
        """Calcola il fitness score per un gene"""
        if metrics["total_trades"] < self.min_trades:
            return 0
            
        # Calcola profit score
        profit_score = (
            min(1, metrics["total_pnl"] / initial_capital) * self.weights["profit_score"]["total_pnl"] +
            (1 - metrics["max_drawdown"]) * self.weights["profit_score"]["max_drawdown"] +
            max(0, metrics["sharpe_ratio"]) * self.weights["profit_score"]["sharpe_ratio"]
        )
        
        # Calcola quality score
        quality_score = (
            metrics["win_rate"] * self.weights["quality_score"]["win_rate"] +
            min(1, metrics["total_trades"] / 100) * self.weights["quality_score"]["trade_frequency"]
        )
        
        # Score finale
        return (profit_score * self.weights["final"]["profit"] + 
                quality_score * self.weights["final"]["quality"])

    def create_next_generation(self, evaluated_population: List[Tuple[TradingGene, float]]):
        """Crea la prossima generazione di geni"""
        new_population = [gene for gene, _ in evaluated_population[:self.elite_size]]
        
        while len(new_population) < self.population_size:
            if np.random.random() < 0.8:  # 80% crossover
                parent1 = self.select_parent(evaluated_population)
                parent2 = self.select_parent(evaluated_population)
                child = parent1.crossover(parent2)
                child.mutate(self.mutation_rate)
            else:  # 20% nuovi geni casuali
                child = TradingGene(random_init=True)
            new_population.append(child)
        
        self.population = new_population

    def select_parent(self, evaluated_population: List[Tuple[TradingGene, float]]) -> TradingGene:
        """Seleziona un genitore usando tournament selection"""
        tournament = np.random.choice(len(evaluated_population), 
                                    size=self.tournament_size, 
                                    replace=False)
        tournament_pop = [evaluated_population[i] for i in tournament]
        return max(tournament_pop, key=lambda x: x[1])[0]

    def optimize(self, simulator: TradingSimulator) -> Tuple[TradingGene, Dict]:
        """Esegue l'ottimizzazione genetica completa"""
        print("Inizializzazione popolazione...")
        self.population = [TradingGene(random_init=True) 
                         for _ in range(self.population_size)]
        
        best_fitness = float('-inf')
        generations_without_improvement = 0
        best_gene = None
        
        for generation in range(self.generations):
            generation_start = datetime.now()
            
            # Valuta popolazione
            evaluated_population = self.evaluate_population(simulator)
            current_best = evaluated_population[0]
            
            # Aggiorna migliori risultati
            if current_best[1] > best_fitness:
                best_fitness = current_best[1]
                best_gene = current_best[0]
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
            
            # Early stopping
            if generations_without_improvement >= 10:
                break
                
            # Crea prossima generazione
            self.create_next_generation(evaluated_population)
        
        return best_gene, {
            'best_fitness': best_fitness,
            'generation_stats': self.generation_stats
        }