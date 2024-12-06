# src/optimization/genetic.py
import multiprocessing
import logging
import traceback
from typing import List, Tuple, Dict
import numpy as np
from copy import deepcopy

from ..models.genes.base import TradingGene
from ..models.genes.volatility import VolatilityAdaptiveGene
from ..models.genes.momentum import MomentumGene 
from ..models.genes.pattern import PatternRecognitionGene
from ..models.simulator import TradingSimulator
from ..utils.config import config

# Configurazione logging dettagliato
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    
)
logger = logging.getLogger(__name__)

def init_worker():
    """Inizializza il worker process"""
    logger.info(f"Worker {multiprocessing.current_process().name} inizializzato")

class ParallelGeneticOptimizer:
    def __init__(self):
        self.population_size = config.get("genetic.population_size", 100)
        self.generations = config.get("genetic.generations", 50)
        self.mutation_rate = config.get("genetic.mutation_rate", 0.1)
        self.elite_size = config.get("genetic.elite_size", 10)
        self.tournament_size = config.get("genetic.tournament_size", 5)
        self.num_processes = config.get("genetic.parallel_processes", 10) 
        self.batch_size = config.get("genetic.batch_size", 32)
        self.generation_stats = []
        self.population = []
        
        logger.info("Inizializzato optimizer genetico:")
        logger.info(f"Population size: {self.population_size}")
        logger.info(f"Generations: {self.generations}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Processes: {self.num_processes}")

    def evaluate_gene_parallel(self, args: Tuple[TradingGene, TradingSimulator]) -> Tuple[TradingGene, float]:
        """Valuta un gene in parallelo"""
        gene, simulator = args
        process_id = multiprocessing.current_process().name
        
        try:
            logger.info(f"[{process_id}] Iniziata valutazione gene")
            simulator.run_simulation(gene)
            metrics = simulator.get_performance_metrics()
            fitness = self.calculate_fitness(metrics)
            gene.fitness_score = fitness
            logger.info(f"[{process_id}] Gene valutato: fitness={fitness:.4f}")
            return gene, fitness
            
        except Exception as e:
            logger.error(f"[{process_id}] Errore valutazione: {str(e)}")
            logger.error(f"[{process_id}] Traceback: {traceback.format_exc()}")
            return gene, 0.0

    def evaluate_population(self, simulator: TradingSimulator) -> List[Tuple[TradingGene, float]]:
        """Valuta popolazione in parallelo"""
        logger.info(f"Avvio valutazione popolazione: {len(self.population)} geni")
        
        with multiprocessing.Pool(processes=self.num_processes, initializer=init_worker) as pool:
            evaluated_population = []
            current_batch = []
            
            for i, gene in enumerate(self.population):
                current_batch.append((gene, simulator))
                
                if len(current_batch) >= self.batch_size:
                    batch_num = i // self.batch_size + 1
                    logger.info(f"Processo batch {batch_num} ({len(current_batch)} geni)")
                    
                    try:
                        results = pool.map(self.evaluate_gene_parallel, current_batch)
                        evaluated_population.extend(results)
                        logger.info(f"Batch {batch_num} completato. Totale valutati: {len(evaluated_population)}")
                        
                    except Exception as e:
                        logger.error(f"Errore batch {batch_num}: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        raise
                        
                    current_batch = []
            
            if current_batch:
                logger.info(f"Processo batch finale ({len(current_batch)} geni)")
                try:
                    results = pool.map(self.evaluate_gene_parallel, current_batch)
                    evaluated_population.extend(results)
                except Exception as e:
                    logger.error(f"Errore batch finale: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise

        logger.info("Valutazione popolazione completata")
        return evaluated_population

    def calculate_fitness(self, metrics: Dict) -> float:
        """Calcola fitness dalle metriche"""
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
        
        fitness = (
            final_weights.get("profit", 0.6) * profit_score +
            final_weights.get("quality", 0.4) * quality_score
        )
        
        return max(0.0, fitness)

    def tournament_selection(self, population: List[TradingGene], 
                           fitness_scores: List[float]) -> TradingGene:
        tournament_idx = np.random.choice(len(population), 
                                        size=self.tournament_size, 
                                        replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def optimize(self, simulator: TradingSimulator) -> Tuple[TradingGene, Dict]:
        """Esegue ottimizzazione genetica"""
        logger.info("Avvio ottimizzazione genetica")
        
        self.population = [TradingGene(random_init=True) 
                         for _ in range(self.population_size)]
        
        best_fitness = float('-inf')
        best_gene = None
        
        for generation in range(self.generations):
            logger.info(f"Generazione {generation + 1}/{self.generations}")
            
            try:
                evaluated_population = self.evaluate_population(simulator)
                current_population = [gene for gene, _ in evaluated_population]
                fitness_scores = [score for _, score in evaluated_population]
            except Exception as e:
                logger.error(f"Errore fatale generazione {generation + 1}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
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
            
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_gene = deepcopy(current_population[current_best_idx])
            
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elite = [deepcopy(current_population[i]) for i in elite_indices]
            
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(current_population, fitness_scores)
                parent2 = self.tournament_selection(current_population, fitness_scores)
                
                child = parent1.crossover(parent2)
                child.mutate(self.mutation_rate)
                new_population.append(child)
            
            self.population = new_population
            
            # Early stopping
            if generation > 10 and np.std([s["best_fitness"] 
                                         for s in self.generation_stats[-10:]]) < 1e-6:
                logger.info("Early stopping - convergenza raggiunta")
                break
        
        logger.info("\nOttimizzazione completata!")
        logger.info(f"Miglior fitness: {best_fitness:.4f}")
        
        return best_gene, {
            "best_fitness": best_fitness,
            "generations": len(self.generation_stats),
            "population_size": self.population_size
        }