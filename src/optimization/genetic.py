# src/optimization/genetic.py
import multiprocessing
import logging
import traceback
from typing import List, Tuple, Dict
import numpy as np
from copy import deepcopy
import talib
from datetime import datetime

from src.models.common import SignalType
from ..models.genes.base import TradingGene
from ..models.simulator import MarketState, TradingSimulator
from ..utils.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParallelGeneticOptimizer:
    
    def __init__(self):
        # Carica parametri dalla configurazione
        self.population_size = config.get("genetic.population_size", 300)
        self.generations = config.get("genetic.generations", 100)
        self.mutation_rate = config.get("genetic.mutation_rate", 0.2)
        self.elite_size = config.get("genetic.elite_size", 5)
        self.tournament_size = config.get("genetic.tournament_size", 5)
        self.min_trades = config.get("genetic.min_trades", 50)
        self.num_processes = min(
            config.get("genetic.parallel_processes", 10),
            multiprocessing.cpu_count()
        )
        self.batch_size = config.get("genetic.batch_size", 32)
        
        # Inizializza stati
        self.generation_stats = []
        self.population = []
        self.precalculated_data = None
        self.best_gene = None
        self.best_fitness = float('-inf')
        
        logger.info("Initialized genetic optimizer with:")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Generations: {self.generations}")
        logger.info(f"Mutation rate: {self.mutation_rate}")
        logger.info(f"Elite size: {self.elite_size}")
        logger.info(f"Tournament size: {self.tournament_size}")
        logger.info(f"Min trades: {self.min_trades}")
        logger.info(f"Parallel processes: {self.num_processes}")
        logger.info(f"Batch size: {self.batch_size}")

    def _precalculate_indicators(self, market_state: MarketState) -> Dict[str, np.ndarray]:
        """Precalcola indicatori tecnici comuni"""
        logger.info("Precalculating technical indicators...")
        
        indicators = {}
        try:
            # Usa gli stessi periodi definiti in TradingGene
            periods = TradingGene.VALID_PERIODS
            
            # Log dei periodi che verranno calcolati
            logger.info(f"Calculating indicators for periods: {periods}")
            
            for period in periods:
                # SMA
                sma = talib.SMA(market_state.close, timeperiod=period)
                indicators[f"SMA_{period}"] = sma
                
                # EMA
                ema = talib.EMA(market_state.close, timeperiod=period)
                indicators[f"EMA_{period}"] = ema
                
                # RSI
                rsi = talib.RSI(market_state.close, timeperiod=period)
                indicators[f"RSI_{period}"] = rsi
                
                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(
                    market_state.close, 
                    timeperiod=period,
                    nbdevup=2,
                    nbdevdn=2
                )
                indicators[f"BB_UPPER_{period}"] = upper
                indicators[f"BB_LOWER_{period}"] = lower

            # Aggiungi dati grezzi
            indicators["CLOSE"] = market_state.close

            # Log degli indicatori disponibili
            logger.info(f"Precalculated {len(indicators)} indicators:")
            logger.info(f"Available indicators: {sorted(indicators.keys())}")

            return indicators
            
        except Exception as e:
            logger.error(f"Error precalculating indicators: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _initialize_population(self) -> List[TradingGene]:
        """Inizializza popolazione con diversità garantita"""
        logger.info("Initializing genetic population...")
        population = []
        
        # Crea geni con diversi tipi di strategie
        for _ in range(self.population_size):
            gene = TradingGene(random_init=True)
            population.append(gene)
        
        logger.info(f"Successfully initialized {len(population)} genes")
        return population

    def _selection_and_reproduction(self, population: List[TradingGene], 
                                  fitness_scores: List[float]) -> List[TradingGene]:
        """Esegue selezione e riproduzione della popolazione"""
        try:
            # Seleziona elite
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            new_population = [deepcopy(population[i]) for i in elite_indices]
            
            # Calcola probabilità di selezione basate sul fitness
            selection_probs = np.array(fitness_scores)
            selection_probs = np.exp(selection_probs - np.max(selection_probs))
            selection_probs = selection_probs / selection_probs.sum()
            
            # Tournament selection e crossover
            while len(new_population) < self.population_size:
                if np.random.random() < 0.8:  # 80% crossover, 20% mutazione
                    # Tournament selection
                    parent1 = self._tournament_selection(population, selection_probs)
                    parent2 = self._tournament_selection(population, selection_probs)
                    
                    # Crossover
                    child = parent1.crossover(parent2)
                    
                    # Adaptive mutation
                    generation_progress = len(self.generation_stats) / self.generations
                    adaptive_rate = self.mutation_rate * (1 - generation_progress * 0.5)
                    child.mutate(adaptive_rate)
                else:
                    # Mutate existing gene
                    parent = self._tournament_selection(population, selection_probs)
                    child = deepcopy(parent)
                    child.mutate(self.mutation_rate * 2)  # Higher mutation rate
                
                new_population.append(child)
            
            return new_population
            
        except Exception as e:
            logger.error(f"Error in selection and reproduction: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _tournament_selection(self, population: List[TradingGene], 
                            selection_probs: np.ndarray) -> TradingGene:
        """Tournament selection with fitness-proportional probability"""
        tournament_idx = np.random.choice(
            len(population),
            size=self.tournament_size,
            replace=False
        )
        
        tournament_probs = selection_probs[tournament_idx]
        tournament_probs = tournament_probs / tournament_probs.sum()
        
        winner_idx = np.random.choice(tournament_idx, p=tournament_probs)
        return population[winner_idx]

    def _evaluate_population(self, simulator: TradingSimulator) -> Tuple[List[float], float, TradingGene]:
        """Valuta la popolazione corrente in parallelo"""
        try:
            fitness_scores = []
            best_generation_fitness = float('-inf')
            best_generation_gene = None
            
            # Valutazione in batch paralleli
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                for i in range(0, len(self.population), self.batch_size):
                    batch = [(gene, simulator) for gene in 
                            self.population[i:i + self.batch_size]]
                    
                    batch_results = pool.map_async(
                        self._evaluate_gene_parallel, batch
                    )
                    
                    for gene, fitness in batch_results.get():
                        fitness_scores.append(fitness)
                        
                        if fitness > best_generation_fitness:
                            best_generation_fitness = fitness
                            best_generation_gene = deepcopy(gene)
            
            return fitness_scores, best_generation_fitness, best_generation_gene
            
        except Exception as e:
            logger.error(f"Error in population evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _evaluate_gene_parallel(self, args: Tuple[TradingGene, TradingSimulator]) -> Tuple[TradingGene, float]:
        """Valutazione parallela di un singolo gene"""
        gene, simulator = args
        process_id = multiprocessing.current_process().name
        
        try:
            # Generate entry conditions
            entry_conditions = gene.generate_entry_conditions(self.precalculated_data)
            
            # Run simulation
            metrics = simulator.run_simulation_vectorized(entry_conditions)
            
            # Calculate fitness
            fitness = self._calculate_fitness(metrics)
            gene.fitness_score = fitness
            
            return gene, fitness
            
        except Exception as e:
            logger.error(f"[{process_id}] Error evaluating gene: {str(e)}")
            logger.error(traceback.format_exc())
            return gene, 0.0


# ... (resto del codice rimane uguale fino alla funzione optimize)

    def _calculate_diversity(self) -> int:
        """Calcola la diversità della popolazione in modo sicuro"""
        try:
            # Creiamo una rappresentazione hashable del DNA di ogni gene
            gene_signatures = set()
            for gene in self.population:
                # Creiamo una lista di tuple con i valori del DNA
                signature = []
                for key, value in sorted(gene.dna.items()):
                    if isinstance(value, dict):
                        # Convertiamo il dizionario in una lista di tuple
                        dict_items = sorted(value.items())
                        signature.append((key, tuple(dict_items)))
                    else:
                        signature.append((key, value))
                gene_signatures.add(tuple(signature))
            
            return len(gene_signatures)
        except Exception as e:
            logger.error(f"Error calculating diversity: {str(e)}")
            return 0

    def optimize(self, simulator: TradingSimulator) -> Tuple[TradingGene, Dict]:
        """Esegue l'ottimizzazione genetica completa"""
        try:
            logger.info("Starting genetic optimization")
            start_time = datetime.now()
            
            # Precalculate indicators
            self.precalculated_data = self._precalculate_indicators(simulator.market_state)
            
            # Initialize population
            self.population = self._initialize_population()
            
            # Evolution loop
            generations_without_improvement = 0
            
            for generation in range(self.generations):
                generation_start = datetime.now()
                logger.info(f"\nGeneration {generation + 1}/{self.generations}")
                
                # Evaluate population
                fitness_scores, best_gen_fitness, best_gen_gene = self._evaluate_population(simulator)
                
                # Update best gene if improved
                if best_gen_fitness > self.best_fitness:
                    self.best_fitness = best_gen_fitness
                    self.best_gene = deepcopy(best_gen_gene)
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Calculate generation statistics
                population_diversity = self._calculate_diversity()
                gen_stats = {
                    "generation": generation + 1,
                    "best_fitness": best_gen_fitness,
                    "avg_fitness": np.mean(fitness_scores),
                    "std_fitness": np.std(fitness_scores),
                    "population_diversity": population_diversity,
                    "elapsed_time": (datetime.now() - generation_start).total_seconds()
                }
                self.generation_stats.append(gen_stats)
                
                # Log progress
                logger.info(f"Best Fitness: {gen_stats['best_fitness']:.4f}")
                logger.info(f"Average Fitness: {gen_stats['avg_fitness']:.4f}")
                logger.info(f"Population Diversity: {gen_stats['population_diversity']}")
                logger.info(f"Time: {gen_stats['elapsed_time']:.2f}s")
                
                # Check early stopping conditions
                if generations_without_improvement >= 20:
                    logger.info("Early stopping - No improvement for 20 generations")
                    break
                    
                # Create next generation
                if generation < self.generations - 1:  # Skip if last generation
                    self.population = self._selection_and_reproduction(
                        self.population, fitness_scores
                    )
                    
                    # Add random genes if diversity is low
                    unique_genes = population_diversity
                    if unique_genes < self.population_size * 0.5:
                        num_random = int(self.population_size * 0.1)
                        logger.info(f"Adding {num_random} random genes to maintain diversity")
                        self.population = self.population[:-num_random]
                        self.population.extend(self._initialize_population()[:num_random])
            
            # Calculate final statistics
            total_time = (datetime.now() - start_time).total_seconds()
            final_stats = {
                "best_fitness": self.best_fitness,
                "generations": len(self.generation_stats),
                "total_time": total_time,
                "avg_generation_time": total_time / len(self.generation_stats),
                "early_stopped": generations_without_improvement >= 20,
                "final_population_size": len(self.population),
                "final_diversity": self._calculate_diversity()
            }
            
            logger.info("\nOptimization completed!")
            logger.info(f"Best fitness achieved: {self.best_fitness:.4f}")
            logger.info(f"Total generations: {final_stats['generations']}")
            logger.info(f"Total time: {final_stats['total_time']:.2f}s")
            
            return self.best_gene, final_stats
            
        except Exception as e:
            logger.error("Error in genetic optimization:")
            logger.error(str(e))
            logger.error(traceback.format_exc())
            raise






    def _calculate_fitness(self, metrics: Dict) -> float:
        """Calcola il fitness score del gene"""
        # Check minimum requirements
        if metrics["total_trades"] < self.min_trades:
            return 0.0
            
        try:
            weights = config.get("genetic.fitness_weights", {})
            profit_weights = weights.get("profit_score", {})
            quality_weights = weights.get("quality_score", {})
            final_weights = weights.get("final_weights", {})
            
            # Profit score components
            profit_score = (
                profit_weights.get("total_pnl", 0.4) * metrics["total_pnl"] / 10000 +
                profit_weights.get("max_drawdown", 0.3) * (1 - metrics["max_drawdown"]) +
                profit_weights.get("sharpe_ratio", 0.3) * max(0, metrics["sharpe_ratio"]) / 3
            )
            
            # Quality score components
            quality_score = (
                quality_weights.get("win_rate", 0.6) * metrics["win_rate"] +
                quality_weights.get("trade_frequency", 0.4) * min(1.0, metrics["total_trades"] / 100) +
                0.2 * (metrics["profit_factor"] - 1) / 2  # Bonus for profit factor > 1
            )
            
            # Apply penalties
            penalties = 1.0
            if metrics["total_trades"] > 500:  # Too many trades
                penalties *= 0.8
            if metrics["max_drawdown"] > 0.3:  # High drawdown
                penalties *= 0.7
            if metrics["win_rate"] < 0.4:  # Poor win rate
                penalties *= 0.9
            
            # Calculate final score
            final_score = (
                final_weights.get("profit", 0.6) * profit_score +
                final_weights.get("quality", 0.4) * quality_score
            ) * penalties
            
            return max(0.0, final_score)
            
        except Exception as e:
            logger.error(f"Error calculating fitness: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def get_generation_stats(self) -> List[Dict]:
        """Restituisce le statistiche complete dell'evoluzione"""
        return self.generation_stats

    def get_best_gene(self) -> TradingGene:
        """Restituisce il miglior gene trovato"""
        return deepcopy(self.best_gene)

if __name__ == "__main__":
    # Test code
    import pandas as pd
    from src.utils.data import load_and_prepare_data
    from src.models.common import TimeFrame
    
    # Load test data
    data_file = "market_data_BTC_1m.csv"
    market_data = load_and_prepare_data(data_file)
    
    # Initialize simulator
    simulator = TradingSimulator()
    simulator.add_market_data(TimeFrame.M1, market_data['1m'])
    
    # Run optimization
    optimizer = ParallelGeneticOptimizer()
    best_gene, stats = optimizer.optimize(simulator)
    
    print("\nOptimization Results:")
    print(f"Best Fitness: {stats['best_fitness']:.4f}")
    print(f"Generations: {stats['generations']}")
    print(f"Total Time: {stats['total_time']:.2f}s")
    print("\nBest Gene Parameters:")
    for key, value in best_gene.dna.items():
        print(f"{key}: {value}")