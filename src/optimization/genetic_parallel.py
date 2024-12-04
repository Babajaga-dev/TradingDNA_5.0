from multiprocessing import Pool, cpu_count
import numpy as np
from typing import List, Tuple, Dict

import pandas as pd
from src.models.gene import TradingGene
from src.models.simulator import TradingSimulator, TimeFrame
from src.utils.config import config

class ParallelGeneticOptimizer:
    def __init__(self):
        self.population_size = config.get("genetic.population_size", 100)
        self.generations = config.get("genetic.generations", 50)
        self.mutation_rate = config.get("genetic.mutation_rate", 0.1)
        self.elite_size = config.get("genetic.elite_size", 10)
        self.tournament_size = config.get("genetic.tournament_size", 5)
        self.min_trades = config.get("genetic.min_trades", 10)
        self.num_processes = cpu_count()  # Utilizza tutti i core disponibili
        
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
        self.best_gene = None
        self.generation_stats = []

    def evaluate_gene_parallel(self, args) -> Tuple[TradingGene, float]:
        """Funzione wrapper per la valutazione parallela dei geni"""
        gene, simulator = args
        simulator.run_simulation(gene)
        metrics = simulator.get_performance_metrics()
        fitness = self.calculate_fitness(metrics, simulator.initial_capital)
        gene.fitness_score = fitness
        gene.performance_history = metrics
        return gene, fitness

    def evaluate_population(self, simulator: TradingSimulator) -> List[Tuple[TradingGene, float]]:
        """Valuta la popolazione in parallelo"""
        print(f"\nValutazione popolazione di {len(self.population)} geni usando {self.num_processes} processi...")
        
        # Crea copie del simulatore per ogni processo
        simulators = [TradingSimulator() for _ in range(len(self.population))]
        for sim in simulators:
            for timeframe, data in simulator.market_data.items():
                sim.add_market_data(timeframe, data)
        
        # Prepara gli argomenti per la valutazione parallela
        eval_args = list(zip(self.population, simulators))
        
        # Esegui la valutazione in parallelo
        with Pool(processes=self.num_processes) as pool:
            results = []
            for i, result in enumerate(pool.imap_unordered(self.evaluate_gene_parallel, eval_args)):
                results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"Valutati {i + 1}/{len(self.population)} geni")
                    if i + 1 < len(self.population):
                        gene, fitness = result
                        metrics = gene.performance_history
                        print(f"  Trade totali: {metrics['total_trades']}")
                        print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
                        print(f"  P&L: ${metrics['total_pnl']:.2f}")
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def create_next_generation_parallel(self, evaluated_population: List[Tuple[TradingGene, float]]):
        """Crea la prossima generazione usando operazioni parallele dove possibile"""
        new_population = [gene for gene, _ in evaluated_population[:self.elite_size]]
        
        num_children = self.population_size - len(new_population)
        parents = self.select_parents(evaluated_population, num_children)
        
        # Prepara gli argomenti per la creazione parallela dei figli
        parent_pairs = [(parents[i], parents[i+1]) for i in range(0, len(parents)-1, 2)]
        
        with Pool(processes=self.num_processes) as pool:
            # Funzione locale per il crossover e la mutazione
            def create_child(parent_pair):
                parent1, parent2 = parent_pair
                child = parent1.crossover(parent2)
                child.mutate(self.mutation_rate)
                return child
            
            # Crea i figli in parallelo
            children = pool.map(create_child, parent_pairs)
            new_population.extend(children)
        
        # Aggiungi geni random se necessario per completare la popolazione
        while len(new_population) < self.population_size:
            new_gene = TradingGene()
            new_gene.mutate(1.0)  # Mutazione completa per massima diversità
            new_population.append(new_gene)
        
        self.population = new_population

    def optimize(self, simulator: TradingSimulator) -> TradingGene:
        """Esegue l'ottimizzazione genetica completa con parallelizzazione"""
        print("\n" + "="*50)
        print(f"OTTIMIZZAZIONE GENETICA PARALLELA ({self.num_processes} processi)")
        print("="*50)
        
        print("\nInizializzazione popolazione...")
        self.population = [TradingGene() for _ in range(self.population_size)]
        
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        for generation in range(self.generations):
            print(f"\n{'='*50}")
            print(f"Generazione {generation + 1}/{self.generations}")
            print(f"{'='*50}")
            
            evaluated_population = self.evaluate_population(simulator)
            current_best_fitness = evaluated_population[0][1]
            current_best_gene = evaluated_population[0][0]
            
            if current_best_fitness > best_fitness:
                improvement = ((current_best_fitness - best_fitness) / abs(best_fitness)) * 100 if best_fitness != float('-inf') else float('inf')
                print(f"\n🎯 Nuovo miglior gene trovato! Miglioramento: {improvement:.1f}%")
                best_fitness = current_best_fitness
                self.best_gene = current_best_gene
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                print(f"\nGenerazioni senza miglioramenti: {generations_without_improvement}")
            
            # Aggiorna le statistiche
            avg_fitness = np.mean([fitness for _, fitness in evaluated_population])
            self.generation_stats.append({
                'generation': generation + 1,
                'best_fitness': current_best_fitness,
                'avg_fitness': avg_fitness,
                'best_pnl': current_best_gene.performance_history['total_pnl']
            })
            
            print(f"\nStatistiche generazione:")
            print(f"  Miglior Fitness: {current_best_fitness:.4f}")
            print(f"  Fitness Media: {avg_fitness:.4f}")
            print(f"  Miglior P&L: ${current_best_gene.performance_history['total_pnl']:.2f}")
            
            if generations_without_improvement >= 10:
                print("\n⚠️ Ottimizzazione terminata per mancanza di miglioramenti")
                break
            
            self.create_next_generation_parallel(evaluated_population)
        
        return self.best_gene

def run_parallel_genetic_trading_system(market_data: pd.DataFrame, 
                                      timeframe: TimeFrame = TimeFrame.M1):
    """Funzione principale per eseguire il sistema di trading genetico parallelizzato"""
    simulator = TradingSimulator()
    simulator.add_market_data(timeframe, market_data)
    optimizer = ParallelGeneticOptimizer()
    return optimizer.optimize(simulator), optimizer