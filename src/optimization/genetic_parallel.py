from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from src.models.gene import TradingGene
from src.models.simulator import TradingSimulator, TimeFrame
from src.utils.config import config
import random
import time


from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from src.models.gene import TradingGene
from src.models.simulator import TradingSimulator, TimeFrame
from src.utils.config import config
import random
import time

class ParallelGeneticOptimizer:


    def __init__(self):
        self.population_size = config.get("genetic.population_size", 100)
        self.generations = config.get("genetic.generations", 50)
        self.mutation_rate = config.get("genetic.mutation_rate", 0.1)
        self.elite_size = config.get("genetic.elite_size", 10)
        self.tournament_size = config.get("genetic.tournament_size", 5)
        self.min_trades = config.get("genetic.min_trades", 10)
        
        # Gestione processi paralleli
        configured_processes = config.get("genetic.parallel_processes", 0)
        if configured_processes > 0:
            self.num_processes = configured_processes
        else:
            # Se impostato a 0 o non specificato, usa cpu_count con limite di 8
            self.num_processes = min(8, cpu_count())
            
        self.batch_size = config.get("genetic.batch_size", 10)
        
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
        
        # Logging configurazione
        print("\n" + "="*50)
        print("CONFIGURAZIONE OTTIMIZZATORE GENETICO")
        print("="*50)
        print(f"Popolazione: {self.population_size}")
        print(f"Generazioni: {self.generations}")
        print(f"Tasso mutazione: {self.mutation_rate*100}%")
        print(f"Elite size: {self.elite_size}")
        print(f"Tournament size: {self.tournament_size}")
        print(f"Min trades: {self.min_trades}")
        print(f"Processi paralleli: {self.num_processes}")
        print(f"Dimensione batch: {self.batch_size}")
        print("\nPESI FITNESS:")
        print("  Profit Score:")
        print(f"    Total PNL: {self.weights['profit_score']['total_pnl']}")
        print(f"    Max Drawdown: {self.weights['profit_score']['max_drawdown']}")
        print(f"    Sharpe Ratio: {self.weights['profit_score']['sharpe_ratio']}")
        print("  Quality Score:")
        print(f"    Win Rate: {self.weights['quality_score']['win_rate']}")
        print(f"    Trade Frequency: {self.weights['quality_score']['trade_frequency']}")
        print("  Final Weights:")
        print(f"    Profit: {self.weights['final']['profit']}")
        print(f"    Quality: {self.weights['final']['quality']}")

    def print_gene_metrics(self, gene_idx: int, gene: TradingGene, metrics: Dict):
        """Stampa le metriche dettagliate per un gene"""
        print(f"\n{'>'*20} Gene {gene_idx + 1} {'<'*20}")
        print("DNA Corrente:")
        for key, value in gene.dna.items():
            print(f"  {key}: {value}")
            
        print(f"\nRisultati Gene {gene_idx + 1}:")
        print(f"  Trade totali: {metrics['total_trades']}")
        print(f"  Trade vincenti: {metrics['winning_trades']}")
        print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"  P&L: ${metrics['total_pnl']:.2f}")
        print(f"  Capitale finale: ${metrics['final_capital']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    def evaluate_gene_parallel(self, args) -> Tuple[TradingGene, float]:
        """Funzione wrapper per la valutazione parallela dei geni"""
        try:
            gene_idx, gene, market_data_dict = args
            start_time = time.time()
            
            simulator = TradingSimulator()
            
            for timeframe_str, data in market_data_dict.items():
                timeframe = TimeFrame(timeframe_str)
                df = pd.DataFrame(data)
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                simulator.add_market_data(timeframe, df)
            
            simulator.run_simulation(gene)
            metrics = simulator.get_performance_metrics()
            fitness = self.calculate_fitness(metrics, simulator.initial_capital)
            gene.fitness_score = fitness
            gene.performance_history = metrics
            
            elapsed_time = time.time() - start_time
            self.print_gene_metrics(gene_idx, gene, metrics)
            print(f"Tempo di valutazione: {elapsed_time:.2f} secondi")
            
            return gene, fitness
            
        except Exception as e:
            print(f"Errore durante la valutazione del gene {gene_idx}: {str(e)}")
            raise


    def print_gene_metrics(self, gene_idx: int, gene: TradingGene, metrics: Dict):
        """Stampa le metriche dettagliate per un gene in modo formattato"""
        # Stampa separatore
        print(f"\n{'='*80}")
        print(f"VALUTAZIONE GENE {gene_idx + 1}")
        print(f"{'='*80}")

        # Stampa DNA
        print("\nDNA:")
        for key, value in gene.dna.items():
            print(f"  {key}: {value}")
        
        # Stampa metriche
        print("\nMETRICHE DI PERFORMANCE:")
        print(f"  Trade totali: {metrics['total_trades']}")
        print(f"  Trade vincenti: {metrics['winning_trades']}")
        print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"  P&L: ${metrics['total_pnl']:.2f}")
        print(f"  Capitale finale: ${metrics['final_capital']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        # Stampa separatore finale
        print(f"\n{'-'*80}")

    def evaluate_population(self, simulator: TradingSimulator) -> List[Tuple[TradingGene, float]]:
        """Valuta la popolazione in parallelo"""
        print(f"\n{'='*80}")
        print(f"VALUTAZIONE POPOLAZIONE - {len(self.population)} GENI")
        print(f"{'='*80}")
        print(f"Utilizzo {self.num_processes} processi in parallelo")
        
        market_data_dict = {}
        for timeframe, data in simulator.market_data.items():
            if isinstance(data, pd.DataFrame):
                df_dict = data.copy()
                df_dict['timestamp'] = df_dict['timestamp'].astype(str)
                market_data_dict[timeframe.value] = df_dict.to_dict('records')
            else:
                market_data_dict[timeframe.value] = [{
                    'timestamp': d.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': d.open,
                    'high': d.high,
                    'low': d.low,
                    'close': d.close,
                    'volume': d.volume
                } for d in data]
        
        eval_args = [(i, gene, market_data_dict) for i, gene in enumerate(self.population)]
        batch_size = self.batch_size
        results = []
        total_batches = (len(eval_args) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(eval_args))
            current_batch = eval_args[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
            print(f"{'='*40}")
            
            with Pool(processes=self.num_processes) as pool:
                batch_results = list(pool.imap_unordered(self.evaluate_gene_parallel, current_batch))
                results.extend(batch_results)
            
           # time.sleep(0.1)
        
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        
        # Stampa sommario finale
        print(f"\n{'='*80}")
        print("SOMMARIO POPOLAZIONE")
        print(f"{'='*80}")
        
        total_trades = sum(gene.performance_history['total_trades'] for gene, _ in results)
        total_pnl = sum(gene.performance_history['total_pnl'] for gene, _ in results)
        avg_win_rate = np.mean([gene.performance_history['win_rate'] for gene, _ in results])
        
        print("\nStatistiche medie:")
        print(f"  Media trade per gene: {total_trades/len(self.population):.1f}")
        print(f"  Media P&L per gene: ${total_pnl/len(self.population):.2f}")
        print(f"  Win Rate medio: {avg_win_rate*100:.1f}%")
        
        print("\n🌟 MIGLIOR GENE DELLA POPOLAZIONE:")
        best_gene, best_fitness = sorted_results[0]
        self.print_gene_metrics(0, best_gene, best_gene.performance_history)
        
        return sorted_results

    def calculate_fitness(self, metrics: Dict, initial_capital: float) -> float:
        """Calcola il fitness di un gene basato sulle sue performance"""
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
        
        # Calcola score finale
        return (profit_score * self.weights["final"]["profit"] + 
                quality_score * self.weights["final"]["quality"])

    def create_next_generation(self, evaluated_population: List[Tuple[TradingGene, float]]):
        """Crea la prossima generazione (versione non parallela per evitare overhead)"""
        new_population = [gene for gene, _ in evaluated_population[:self.elite_size]]
        
        while len(new_population) < self.population_size:
            if random.random() < 0.8:  # 80% probability of crossover
                parent1 = self.select_parent(evaluated_population)
                parent2 = self.select_parent(evaluated_population)
                child = parent1.crossover(parent2)
                child.mutate(self.mutation_rate)
            else:  # 20% probability of new random gene
                child = TradingGene(random_init=True)
            new_population.append(child)
        
        self.population = new_population

    def select_parent(self, evaluated_population: List[Tuple[TradingGene, float]]) -> TradingGene:
        """Seleziona un genitore usando il metodo del torneo"""
        tournament = random.sample(evaluated_population, self.tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def optimize(self, simulator: TradingSimulator) -> TradingGene:
        """Esegue l'ottimizzazione genetica"""
        print("\n" + "="*50)
        print(f"OTTIMIZZAZIONE GENETICA PARALLELA ({self.num_processes} processi)")
        print("="*50)
        
        print("\nInizializzazione popolazione...")
        self.population = [TradingGene(random_init=True) for _ in range(self.population_size)]
        
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        for generation in range(self.generations):
            generation_start_time = time.time()
            
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
            
            generation_time = time.time() - generation_start_time
            print(f"\nStatistiche generazione {generation + 1}:")
            print(f"  Tempo impiegato: {generation_time:.1f} secondi")
            print(f"  Miglior Fitness: {current_best_fitness:.4f}")
            print(f"  Fitness Media: {avg_fitness:.4f}")
            print(f"  Miglior P&L: ${current_best_gene.performance_history['total_pnl']:.2f}")
            
            if generations_without_improvement >= 10:
                print("\n⚠️ Ottimizzazione terminata per mancanza di miglioramenti")
                break
            
            self.create_next_generation(evaluated_population)
        
        return self.best_gene

def run_parallel_genetic_trading_system(market_data: pd.DataFrame, 
                                      timeframe: TimeFrame = TimeFrame.M1):
    """Funzione principale per eseguire il sistema di trading genetico parallelizzato"""
    simulator = TradingSimulator()
    simulator.add_market_data(timeframe, market_data)
    optimizer = ParallelGeneticOptimizer()
    return optimizer.optimize(simulator), optimizer