import random
import numpy as np
from typing import List, Tuple, Dict
from src.models.gene import TradingGene
from src.models.simulator import TradingSimulator, TimeFrame
from src.utils.config import config
import pandas as pd
from datetime import datetime

class GeneticOptimizer:
    

    def evaluate_population(self, simulator: TradingSimulator) -> List[Tuple[TradingGene, float]]:
        results = []
        print(f"\n{'='*80}")
        print(f"VALUTAZIONE POPOLAZIONE - {len(self.population)} GENI")
        print(f"{'='*80}")
        
        best_metrics = None
        total_trades = 0
        total_pnl = 0
        
        for i, gene in enumerate(self.population, 1):
            print(f"\n{'>'*20} Valutazione Gene {i}/{len(self.population)} {'<'*20}")
            print("DNA Corrente:")
            for key, value in gene.dna.items():
                print(f"  {key}: {value}")
            
            simulator.run_simulation(gene)
            metrics = simulator.get_performance_metrics()
            
            print(f"\nRisultati Gene {i}:")
            print(f"  Trade totali: {metrics['total_trades']}")
            print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"  P&L: ${metrics['total_pnl']:.2f}")
            
            total_trades += metrics['total_trades']
            total_pnl += metrics['total_pnl']
            
            fitness = self.calculate_fitness(metrics, simulator.initial_capital)
            gene.fitness_score = fitness
            gene.performance_history = metrics
            results.append((gene, fitness))
            
            if not best_metrics or metrics['total_pnl'] > best_metrics['total_pnl']:
                best_metrics = metrics
                print("\n🌟 NUOVO MIGLIOR GENE TROVATO!")
        
        print(f"\n{'='*80}")
        print("SOMMARIO POPOLAZIONE:")
        print(f"Media trade per gene: {total_trades/len(self.population):.1f}")
        print(f"Media P&L per gene: ${total_pnl/len(self.population):.2f}")
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def create_next_generation(self, evaluated_population: List[Tuple[TradingGene, float]]):
        print(f"\n{'='*80}")
        print("CREAZIONE NUOVA GENERAZIONE")
        print(f"{'='*80}")
        
        # Preserva i migliori geni (elite)
        new_population = [gene for gene, _ in evaluated_population[:self.elite_size]]
        print(f"\n🏆 Preservati {self.elite_size} geni elite")
        
        # Seleziona i genitori per il crossover
        num_children = self.population_size - len(new_population)
        parents = self.select_parents(evaluated_population, num_children)
        print(f"\n👥 Selezionati {len(parents)} genitori per il crossover")
        
        # Crea nuovi geni attraverso crossover e mutazione
        children_created = 0
        print("\n🧬 Creazione nuovi geni...")
        
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = parent1.crossover(parent2)
            
            print(f"\nCreazione gene figlio {children_created + 1}/{num_children}")
            print("Parents DNA:")
            print("  Parent 1:", {k: v for k, v in parent1.dna.items() if not isinstance(v, dict)})
            print("  Parent 2:", {k: v for k, v in parent2.dna.items() if not isinstance(v, dict)})
            
            print("\nApplicazione mutazione...")
            child.mutate(self.mutation_rate)
            
            print("DNA Risultante:")
            print({k: v for k, v in child.dna.items() if not isinstance(v, dict)})
            
            new_population.append(child)
            children_created += 1
        
        print(f"\n✅ Generazione completata:")
        print(f"  Elite preservati: {self.elite_size}")
        print(f"  Nuovi geni creati: {children_created}")
        
        self.population = new_population

    def optimize(self, simulator: TradingSimulator) -> TradingGene:
        print("\n" + "="*80)
        print("AVVIO OTTIMIZZAZIONE GENETICA")
        print("="*80)
        
        print("\nInitializzazione popolazione iniziale...")
        self.initialize_population()
        
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        for generation in range(self.generations):
            print(f"\n{'#'*80}")
            print(f"GENERAZIONE {generation + 1}/{self.generations}")
            print(f"{'#'*80}")
            
            evaluated_population = self.evaluate_population(simulator)
            current_best_fitness = evaluated_population[0][1]
            current_best_gene = evaluated_population[0][0]
            
            if current_best_fitness > best_fitness:
                improvement = ((current_best_fitness - best_fitness) / abs(best_fitness)) * 100 if best_fitness != float('-inf') else float('inf')
                print(f"\n🎯 NUOVO RECORD! Miglioramento: {improvement:.1f}%")
                best_fitness = current_best_fitness
                self.best_gene = current_best_gene
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                print(f"\n⚠️ Nessun miglioramento. Generazioni stagnanti: {generations_without_improvement}")
            
            if generations_without_improvement >= 10:
                print("\n🛑 Ottimizzazione terminata per stagnazione")
                break
                
            self.create_next_generation(evaluated_population)
            
        return self.best_gene

    def __init__(self):
        # Carica i parametri dal file di configurazione
        self.population_size = config.get("genetic.population_size", 100)
        self.generations = config.get("genetic.generations", 50)
        self.mutation_rate = config.get("genetic.mutation_rate", 0.1)
        self.elite_size = config.get("genetic.elite_size", 10)
        self.tournament_size = config.get("genetic.tournament_size", 5)
        self.min_trades = config.get("genetic.min_trades", 10)
        
        # Pesi per il calcolo del fitness
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
            # Aggiungiamo logging configurazione
        print("\n" + "="*50)
        print("CONFIGURAZIONE OTTIMIZZATORE GENETICO")
        print("="*50)
        print(f"Popolazione: {self.population_size}")
        print(f"Generazioni: {self.generations}")
        print(f"Tasso mutazione: {self.mutation_rate*100}%")
        print(f"Elite size: {self.elite_size}")
        print(f"Tournament size: {self.tournament_size}")
        print(f"Min trades: {self.min_trades}")
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
    
    def initialize_population(self):
        """Inizializza la popolazione con geni casuali"""
        print("\nCreazione popolazione iniziale...")
        self.population = []
        
        # Il primo gene usa la configurazione di default
        self.population.append(TradingGene(random_init=False))
        print("Gene 1: Configurazione di default")
        
        # Il resto della popolazione è casuale
        for i in range(1, self.population_size):
            gene = TradingGene(random_init=True)
            self.population.append(gene)
            print(f"Gene {i+1}: Configurazione casuale generata")
            
        print(f"\nPopolazione iniziale creata: {len(self.population)} geni")   
        
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
    
    def select_parents(self, evaluated_population: List[Tuple[TradingGene, float]], 
                      num_parents: int) -> List[TradingGene]:
        """Seleziona i genitori usando il metodo del torneo"""
        parents = []
        for _ in range(num_parents):
            tournament = random.sample(evaluated_population, self.tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)
        return parents

def run_genetic_trading_system(market_data: pd.DataFrame, 
                             timeframe: TimeFrame = TimeFrame.M1):
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
    simulator = TradingSimulator()
    simulator.add_market_data(timeframe, market_data)
    
    # Verifica che i dati siano stati caricati correttamente
    if not simulator.market_data or timeframe not in simulator.market_data:
        raise ValueError(f"Errore nel caricamento dei dati per il timeframe {timeframe}")
    
    print(f"\nDati caricati correttamente: {len(simulator.market_data[timeframe])} candele")
    
    # Crea e avvia l'ottimizzatore
    optimizer = GeneticOptimizer()
    
    print("\nAvvio ottimizzazione genetica...")
    best_gene = optimizer.optimize(simulator)
    
    return best_gene, optimizer
