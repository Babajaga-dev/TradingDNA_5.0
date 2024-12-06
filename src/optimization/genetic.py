# src/optimization/genetic.py
import multiprocessing
import logging
import traceback
from typing import List, Tuple, Dict
import numpy as np
from copy import deepcopy
import talib

from src.models.common import SignalType

from ..models.genes.base import TradingGene
from ..models.simulator import MarketState, TradingSimulator
from ..utils.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ParallelGeneticOptimizer:
    def __init__(self):
        self.population_size = config.get("genetic.population_size", 100)
        self.generations = config.get("genetic.generations", 50)
        self.mutation_rate = config.get("genetic.mutation_rate", 0.1)
        self.elite_size = config.get("genetic.elite_size", 10)
        self.tournament_size = config.get("genetic.tournament_size", 5)
        self.num_processes = min(
            config.get("genetic.parallel_processes", 10),
            multiprocessing.cpu_count()
        )
        self.batch_size = config.get("genetic.batch_size", 32)
        self.generation_stats = []
        self.population = []
        self.precalculated_data = None

    def _get_indicator(self, indicator_type: str, params: Dict) -> np.ndarray:
        """Recupera indicatore precalcolato"""
        if indicator_type == "SMA":
            period = params.get("timeperiod", 20)
            return self.precalculated_data[f"SMA_{period}"]
        elif indicator_type == "RSI":
            period = params.get("timeperiod", 14) 
            return self.precalculated_data[f"RSI_{period}"]
        elif indicator_type.startswith("BB_"):
            period = params.get("timeperiod", 20)
            return self.precalculated_data[f"{indicator_type}_{period}"]
        elif indicator_type == "CLOSE":
            return self.precalculated_data["CLOSE"]
        else:
            return self.precalculated_data.get(indicator_type, 
                   np.full_like(self.precalculated_data["CLOSE"], np.nan))

    def _selection_and_reproduction(self, population, fitness_scores):
        """Selection and reproduction with elitism"""
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        elite = [deepcopy(population[i]) for i in elite_indices]
        
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            child = parent1.crossover(parent2)
            child.mutate(self.mutation_rate)
            new_population.append(child)
            
        return new_population

    def _tournament_selection(self, population, fitness_scores):
        """Tournament selection"""
        tournament_idx = np.random.choice(
            len(population), 
            size=self.tournament_size,
            replace=False
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def optimize(self, simulator: TradingSimulator) -> Tuple[TradingGene, Dict]:
        logger.info("Starting genetic optimization")
        
        # Precalculate indicators using market_state
        self.precalculated_data = self.precalculate_indicators(simulator.market_state)
        
        self.population = [TradingGene(random_init=True) 
                        for _ in range(self.population_size)]
        
        best_fitness = float('-inf')
        best_gene = None
        
        for generation in range(self.generations):
            logger.info(f"\nGeneration {generation + 1}/{self.generations}")
            try:
                with multiprocessing.Pool(processes=self.num_processes) as pool:
                    results = []
                    for i in range(0, len(self.population), self.batch_size):
                        batch = [(gene, simulator) for gene in 
                                self.population[i:i + self.batch_size]]
                        batch_results = pool.map_async(
                            self.evaluate_gene_parallel, batch
                        )
                        results.extend(batch_results.get())
                
                current_population = [gene for gene, _ in results] 
                fitness_scores = [score for _, score in results]
                
            except Exception as e:
                logger.error(f"Fatal error in generation {generation + 1}: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
            gen_stats = {
                "generation": generation + 1,
                "best_fitness": max(fitness_scores),
                "avg_fitness": np.mean(fitness_scores),
                "std_fitness": np.std(fitness_scores)
            }
            self.generation_stats.append(gen_stats)
            
            logger.info(f"Best Fitness: {gen_stats['best_fitness']:.4f}")
            logger.info(f"Avg Fitness: {gen_stats['avg_fitness']:.4f}")
            
            # Update best gene
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_gene = deepcopy(current_population[current_best_idx])
            
            # Selection and reproduction
            new_population = self._selection_and_reproduction(
                current_population, fitness_scores
            )
            self.population = new_population
            
            # Early stopping check
            if (generation > 10 and 
                np.std([s["best_fitness"] for s in self.generation_stats[-10:]]) < 1e-6):
                logger.info("Early stopping - convergence reached")
                break
        
        logger.info("\nOptimization completed!")
        logger.info(f"Best fitness: {best_fitness:.4f}")
        
        return best_gene, {
            "best_fitness": best_fitness,
            "generations": len(self.generation_stats),
            "population_size": self.population_size
        }

    def precalculate_indicators(self, market_state: MarketState) -> Dict[str, np.ndarray]:
        """Precalcola tutti gli indicatori possibili"""
        logger.info("Precalculating indicators...")
            
        indicators = {}
        common_periods = list(range(5, 200, 5))  # 5, 10, 15, ..., 195
        
        # SMA/EMA per tutti i periodi
        for period in common_periods:
            indicators[f"SMA_{period}"] = talib.SMA(market_state.close, timeperiod=period)
            indicators[f"EMA_{period}"] = talib.EMA(market_state.close, timeperiod=period)
            indicators[f"RSI_{period}"] = talib.RSI(market_state.close, timeperiod=period)
            
            upper, middle, lower = talib.BBANDS(market_state.close, timeperiod=period)
            indicators[f"BB_UPPER_{period}"] = upper
            indicators[f"BB_MIDDLE_{period}"] = middle
            indicators[f"BB_LOWER_{period}"] = lower

        # MACD con varie combinazioni
        for fast in [12, 24, 36]:
            for slow in [26, 52, 78]:
                if slow > fast:
                    macd, signal, hist = talib.MACD(market_state.close, 
                                                fastperiod=fast,
                                                slowperiod=slow,
                                                signalperiod=9)
                    indicators[f"MACD_{fast}_{slow}"] = macd
                    indicators[f"MACD_SIGNAL_{fast}_{slow}"] = signal
                    indicators[f"MACD_HIST_{fast}_{slow}"] = hist

        # Store raw data too
        indicators["CLOSE"] = market_state.close
        indicators["HIGH"] = market_state.high
        indicators["LOW"] = market_state.low
        indicators["OPEN"] = market_state.open

        logger.info(f"Precalculated {len(indicators)} indicators")
        return indicators

    def _calculate_metrics_optimized(self, signals, prices, position_size_pct=0.05):
        if len(signals) == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "final_capital": self.initial_capital,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }

        position_active = np.zeros_like(prices, dtype=bool)
        entry_prices = np.zeros_like(prices)
        pnl = np.zeros_like(prices)
        capital = np.ones_like(prices) * self.initial_capital

        for i in range(1, len(signals)):
            if signals[i] and not position_active[i-1]:  # Entry
                position_active[i:] = True
                entry_prices[i:] = prices[i]
            elif signals[i] and position_active[i-1]:  # Exit
                position_active[i:] = False
                pnl[i] = (prices[i] - entry_prices[i-1]) * position_size_pct
                capital[i] = capital[i-1] + pnl[i]
        
        total_trades = np.sum(np.diff(position_active.astype(int)) != 0) // 2
        winning_trades = np.sum(pnl > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Drawdown
        peaks = np.maximum.accumulate(capital)
        drawdowns = (peaks - capital) / peaks
        max_drawdown = np.max(drawdowns)
        
        # Sharpe
        returns = np.diff(capital) / capital[:-1]
        sharpe = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if len(returns) > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "total_pnl": float(capital[-1] - self.initial_capital),
            "final_capital": float(capital[-1]),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe)
        }

    def evaluate_gene_parallel(self, args: Tuple[TradingGene, TradingSimulator]) -> Tuple[TradingGene, float]:
        gene, simulator = args
        process_id = multiprocessing.current_process().name
        
        try:
            logger.info(f"[{process_id}] Starting gene evaluation")
            
            # Generate entry signals
            entry_conditions = gene.generate_entry_conditions(self.precalculated_data)
            logger.info(f"[{process_id}] Generated entry conditions: Count(True)={np.sum(entry_conditions)}")
            
            # Run simulation
            metrics = simulator.run_simulation_vectorized(entry_conditions)
            logger.info(f"[{process_id}] Simulation metrics: {metrics}")
            
            fitness = self.calculate_fitness(metrics)
            gene.fitness_score = fitness
            
            logger.info(f"[{process_id}] Gene evaluation complete:")
            logger.info(f"[{process_id}] - Total trades: {metrics['total_trades']}")
            logger.info(f"[{process_id}] - Win rate: {metrics['win_rate']:.4f}")
            logger.info(f"[{process_id}] - Total PnL: {metrics['total_pnl']:.4f}")
            logger.info(f"[{process_id}] - Final capital: {metrics['final_capital']:.4f}")
            logger.info(f"[{process_id}] - Max drawdown: {metrics['max_drawdown']:.4f}")
            logger.info(f"[{process_id}] - Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
            logger.info(f"[{process_id}] - Final fitness: {fitness:.4f}")
            
            return gene, fitness
            
        except Exception as e:
            logger.error(f"[{process_id}] Evaluation error: {str(e)}")
            logger.error(traceback.format_exc())
            return gene, 0.0

    def calculate_fitness(self, metrics: Dict) -> float:
        logger.info(f"Starting fitness calculation with metrics: {metrics}")
        
        if metrics["total_trades"] < config.get("genetic.min_trades", 10):
            logger.info(f"Insufficient trades: {metrics['total_trades']} < {config.get('genetic.min_trades', 10)}")
            return 0.0
            
        weights = config.get("genetic.fitness_weights", {})
        profit_weights = weights.get("profit_score", {})
        quality_weights = weights.get("quality_score", {})
        final_weights = weights.get("final_weights", {})
        
        logger.info(f"Weights loaded - profit: {profit_weights}, quality: {quality_weights}, final: {final_weights}")
        
        pnl_score = profit_weights.get("total_pnl", 0.4) * metrics["total_pnl"]
        drawdown_score = profit_weights.get("max_drawdown", 0.3) * (1 - metrics["max_drawdown"])
        sharpe_score = profit_weights.get("sharpe_ratio", 0.3) * max(0, metrics["sharpe_ratio"])
        
        profit_score = pnl_score + drawdown_score + sharpe_score
        
        logger.info(f"Profit scores - PnL: {pnl_score:.4f}, Drawdown: {drawdown_score:.4f}, Sharpe: {sharpe_score:.4f}, Total: {profit_score:.4f}")
        
        winrate_score = quality_weights.get("win_rate", 0.6) * metrics["win_rate"]
        frequency_score = quality_weights.get("trade_frequency", 0.4) * min(1.0, metrics["total_trades"] / 100)
        
        quality_score = winrate_score + frequency_score
        
        logger.info(f"Quality scores - Winrate: {winrate_score:.4f}, Frequency: {frequency_score:.4f}, Total: {quality_score:.4f}")
        
        final_score = max(0.0,
            final_weights.get("profit", 0.6) * profit_score +
            final_weights.get("quality", 0.4) * quality_score
        )
        
        logger.info(f"Final fitness score: {final_score:.4f}")
        
        return final_score

    def _generate_signals_optimized(self, gene):
        try:
            entry_ind1 = self._get_indicator(gene.dna["entry_indicator1"], 
                                        gene.dna["entry_indicator1_params"])
            entry_ind2 = self._get_indicator(gene.dna["entry_indicator2"], 
                                        gene.dna["entry_indicator2_params"])
            
            # Converti entry_conditions in array booleano con numpy.where()
            entry_signals = np.zeros(len(entry_ind1), dtype=bool)
            
            if gene.dna["entry_operator"] == ">":
                entry_signals = np.where(entry_ind1 > entry_ind2, True, False)
            elif gene.dna["entry_operator"] == "<":
                entry_signals = np.where(entry_ind1 < entry_ind2, True, False)
            elif gene.dna["entry_operator"] == "cross_above":
                entry_signals[1:] = (entry_ind1[:-1] <= entry_ind2[:-1]) & (entry_ind1[1:] > entry_ind2[1:])
            elif gene.dna["entry_operator"] == "cross_below":
                entry_signals[1:] = (entry_ind1[:-1] >= entry_ind2[:-1]) & (entry_ind1[1:] < entry_ind2[1:])
            
            # Converti i booleani in SignalType.LONG/EXIT
            signals = np.where(entry_signals, SignalType.LONG, 0)
            
            # Aggiungi segnali di uscita dopo ogni entry
            for i in range(1, len(signals)):
                if signals[i-1] == SignalType.LONG:
                    signals[i] = SignalType.EXIT
                    
            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return np.zeros(len(self.precalculated_data["CLOSE"]), dtype=int)

    def calculate_fitness(self, metrics: Dict) -> float:
        if metrics["total_trades"] < config.get("genetic.min_trades", 10):
            return 0.0
            
        weights = config.get("genetic.fitness_weights", {})
        profit_weights = weights.get("profit_score", {})
        quality_weights = weights.get("quality_score", {})
        final_weights = weights.get("final_weights", {})
        
        profit_score = (
            profit_weights.get("total_pnl", 0.4) * metrics["total_pnl"] +
            profit_weights.get("max_drawdown", 0.3) * (1 - metrics["max_drawdown"]) +
            profit_weights.get("sharpe_ratio", 0.3) * max(0, metrics["sharpe_ratio"])
        )
        
        quality_score = (
            quality_weights.get("win_rate", 0.6) * metrics["win_rate"] +
            quality_weights.get("trade_frequency", 0.4) * min(1.0, metrics["total_trades"] / 100)
        )
        
        return max(0.0, 
            final_weights.get("profit", 0.6) * profit_score +
            final_weights.get("quality", 0.4) * quality_score
        )