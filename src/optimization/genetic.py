import random
import numpy as np
from typing import List, Tuple, Dict
from src.models.gene import TradingGene  # Changed from relative to absolute import
from src.models.simulator import TradingSimulator, TimeFrame  # Changed from relative to absolute import
import pandas as pd

class GeneticOptimizer:

    def initialize_population(self):
        self.population = [TradingGene() for _ in range(self.population_size)]
    
    def calculate_fitness(self, metrics: Dict) -> float:
        """Calcola il fitness di un gene basato sulle sue performance"""
        if metrics["total_trades"] < 10:
            return 0
            
        profit_score = (
            min(1, metrics["total_pnl"] / (metrics["final_capital"] - 10000)) * 0.4 +
            (1 - metrics["max_drawdown"]) * 0.3 +
            max(0, metrics["sharpe_ratio"]) * 0.3
        )
        
        quality_score = (
            metrics["win_rate"] * 0.6 +
            min(1, metrics["total_trades"] / 100) * 0.4
        )
        
        return (profit_score * 0.6 + quality_score * 0.4)
    
    def select_parents(self, evaluated_population: List[Tuple[TradingGene, float]], 
                      num_parents: int) -> List[TradingGene]:
        """Seleziona i genitori usando il metodo del torneo"""
        parents = []
        for _ in range(num_parents):
            tournament = random.sample(evaluated_population, 5)
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)
        return parents
    
    def create_next_generation(self, evaluated_population: List[Tuple[TradingGene, float]]):
        """Crea la prossima generazione di geni"""
        new_population = [gene for gene, _ in evaluated_population[:self.elite_size]]
        
        num_children = self.population_size - len(new_population)
        parents = self.select_parents(evaluated_population, num_children)
        
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = parent1.crossover(parent2)
            child.mutate(self.mutation_rate)
            new_population.append(child)
        
        self.population = new_population

    def __init__(self, 
                 population_size: int = 100, 
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 elite_size: int = 10):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = []
        self.best_gene = None
        self.generation_stats = []
    
    def evaluate_population(self, simulator: TradingSimulator) -> List[Tuple[TradingGene, float]]:
        results = []
        print(f"\nValutazione popolazione di {len(self.population)} geni...")
        
        for i, gene in enumerate(self.population, 1):
            if i % 10 == 0:  # Stampa progresso ogni 10 geni
                print(f"Valutazione gene {i}/{len(self.population)}")
            
            simulator.run_simulation(gene)
            metrics = simulator.get_performance_metrics()
            
            # Stampa metriche del gene corrente
            if i % 10 == 0:
                print(f"  Trade totali: {metrics['total_trades']}")
                print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
                print(f"  P&L: ${metrics['total_pnl']:.2f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
            
            fitness = self.calculate_fitness(metrics)
            gene.fitness_score = fitness
            gene.performance_history = metrics
            results.append((gene, fitness))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def optimize(self, simulator: TradingSimulator) -> TradingGene:
        """Esegue l'ottimizzazione genetica completa"""
        print("\nInizializzazione popolazione...")
        self.initialize_population()
        
        # Stampa informazioni sul dataset
        data_info = simulator.market_data[TimeFrame.M1]
        start_date = data_info[0].timestamp
        end_date = data_info[-1].timestamp
        n_candles = len(data_info)
        print(f"Dataset: dal {start_date} al {end_date}")
        print(f"Numero di candele: {n_candles}")
        
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        for generation in range(self.generations):
            print(f"\n{'='*50}")
            print(f"Generazione {generation + 1}/{self.generations}")
            print(f"{'='*50}")
            
            evaluated_population = self.evaluate_population(simulator)
            current_best_fitness = evaluated_population[0][1]
            current_best_gene = evaluated_population[0][0]
            
            # Stampa dettagli del miglior gene della generazione
            print("\nMiglior gene della generazione:")
            metrics = current_best_gene.performance_history
            print(f"  Trades: {metrics['total_trades']}")
            print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"  P&L: ${metrics['total_pnl']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            
            if current_best_fitness > best_fitness:
                improvement = ((current_best_fitness - best_fitness) / abs(best_fitness)) * 100 if best_fitness != float('-inf') else float('inf')
                print(f"\n🎯 Nuovo miglior gene trovato! Miglioramento: {improvement:.1f}%")
                best_fitness = current_best_fitness
                self.best_gene = current_best_gene
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                print(f"\nGenerazioni senza miglioramenti: {generations_without_improvement}")
            
            avg_fitness = np.mean([fitness for _, fitness in evaluated_population])
            self.generation_stats.append({
                'generation': generation + 1,
                'best_fitness': current_best_fitness,
                'avg_fitness': avg_fitness,
                'best_pnl': metrics['total_pnl']
            })
            
            print(f"\nStatistiche generazione:")
            print(f"  Miglior Fitness: {current_best_fitness:.4f}")
            print(f"  Fitness Media: {avg_fitness:.4f}")
            print(f"  Miglior P&L: ${metrics['total_pnl']:.2f}")
            
            if generations_without_improvement >= 10:
                print("\n⚠️ Ottimizzazione terminata per mancanza di miglioramenti")
                break
            
            self.create_next_generation(evaluated_population)
        
        return self.best_gene


def run_genetic_trading_system(market_data: pd.DataFrame, 
                             timeframe: TimeFrame = TimeFrame.M1,
                             population_size: int = 100,
                             generations: int = 50,
                             initial_capital: float = 10000):
    """Funzione principale per eseguire il sistema di trading genetico"""
    
    print("\nPreparazione dei dati...")
    print(f"Shape dei dati: {market_data.shape}")
    print(f"Colonne disponibili: {market_data.columns.tolist()}")
    print(f"Range date: da {market_data['timestamp'].min()} a {market_data['timestamp'].max()}")
    
    # Verifica che i dati siano nel formato corretto
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in market_data.columns for col in required_columns):
        raise ValueError(f"Mancano alcune colonne richieste. Colonne necessarie: {required_columns}")
    
    # Verifica che timestamp sia in formato datetime
    if not pd.api.types.is_datetime64_any_dtype(market_data['timestamp']):
        market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
    
    # Verifica che i dati numerici siano float
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        market_data[col] = market_data[col].astype(float)
    
    # Inizializza il simulatore
    simulator = TradingSimulator(initial_capital=initial_capital)
    simulator.add_market_data(timeframe, market_data)
    
    # Verifica che i dati siano stati caricati correttamente
    if not simulator.market_data or timeframe not in simulator.market_data:
        raise ValueError(f"Errore nel caricamento dei dati per il timeframe {timeframe}")
    
    print(f"\nDati caricati correttamente: {len(simulator.market_data[timeframe])} candele")
    
    # Crea e avvia l'ottimizzatore
    optimizer = GeneticOptimizer(
        population_size=population_size,
        generations=generations,
        mutation_rate=0.1,
        elite_size=int(population_size * 0.1)
    )
    
    print("\nAvvio ottimizzazione genetica...")
    best_gene = optimizer.optimize(simulator)
    
    return best_gene, optimizer