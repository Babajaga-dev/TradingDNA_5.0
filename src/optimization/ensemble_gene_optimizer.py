from typing import List, Dict, Tuple
import numpy as np
from multiprocessing import Pool
import pandas as pd
from datetime import datetime

import torch
from src.models.common import Signal, SignalType
from src.models.simulator import TradingSimulator, TimeFrame
from src.models.gene import TorchGene, TradingGene
from src.utils.config import config
from src.models.gene import (
    TradingGene, 
    VolatilityAdaptiveGene, 
    MomentumGene, 
    PatternRecognitionGene
)

def run_ensemble_optimization(market_data: pd.DataFrame, 
                            timeframe: TimeFrame = TimeFrame.M1,
                            device: torch.device = None) -> Tuple[List[TorchGene], Dict]:
    """Funzione principale per eseguire l'ottimizzazione dell'ensemble"""
    print("1. Inizializzazione simulatore...")
    simulator = TradingSimulator()
    simulator.add_market_data(timeframe, market_data)
    
    print("2. Setup device...")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device selezionato: {device}")
    
    print("3. Creazione ottimizzatore...")
    optimizer = EnsembleGeneOptimizer(device=device)
    
    print("4. Avvio ottimizzazione...")
    try:
        best_ensemble, stats = optimizer.optimize(simulator)
    except Exception as e:
        print(f"Errore durante l'ottimizzazione: {str(e)}")
        raise
    
    print("\nOttimizzazione completata!")
    return best_ensemble, stats


class EnsembleSignalCombiner:
    """Classe per combinare i segnali da diversi geni dell'ensemble"""
    
    def __init__(self, ensemble_weights: Dict[str, float]):
        self.weights = ensemble_weights
        
    def combine_signals(self, gene_signals: List[List[Signal]]) -> List[Signal]:
        """Combina i segnali dai diversi geni usando un sistema di votazione pesato"""
        if not gene_signals:
            return []
        
        combined_signals = []
        timestamp = gene_signals[0][0].timestamp if gene_signals[0] else None
        
        # Conta i voti pesati per ogni tipo di segnale
        votes = {
            SignalType.LONG: 0,
            SignalType.SHORT: 0,
            SignalType.EXIT: 0
        }
        
        # Raccogli i voti da ogni gene
        total_weight = 0
        for signals, weight in zip(gene_signals, self.weights.values()):
            if signals:
                signal_type = signals[0].type
                votes[signal_type] += weight
                total_weight += weight
        
        if total_weight == 0:
            return []
            
        # Normalizza i voti
        for signal_type in votes:
            votes[signal_type] /= total_weight
            
        # Determina il segnale vincente (richiede almeno 50% dei voti)
        winning_type = max(votes.items(), key=lambda x: x[1])
        if winning_type[1] >= 0.5:
            # Crea un nuovo segnale combinato
            combined_signal = Signal(
                type=winning_type[0],
                timestamp=timestamp,
                price=gene_signals[0][0].price,  # Usa il prezzo dal primo gene
                metadata={"ensemble": True}
            )
            
            # Calcola stop loss e take profit medi pesati
            if winning_type[0] in [SignalType.LONG, SignalType.SHORT]:
                stop_losses = []
                take_profits = []
                
                for signals, weight in zip(gene_signals, self.weights.values()):
                    if signals and signals[0].type == winning_type[0]:
                        if signals[0].stop_loss:
                            stop_losses.append((signals[0].stop_loss, weight))
                        if signals[0].take_profit:
                            take_profits.append((signals[0].take_profit, weight))
                
                if stop_losses:
                    combined_signal.stop_loss = sum(sl * w for sl, w in stop_losses) / sum(w for _, w in stop_losses)
                if take_profits:
                    combined_signal.take_profit = sum(tp * w for tp, w in take_profits) / sum(w for _, w in take_profits)
            
            combined_signals.append(combined_signal)
        
        return combined_signals


class EnsembleGeneOptimizer:
    
    def __init__(self, device: torch.device = None):
        """Inizializza l'ottimizzatore per ensemble di geni"""
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parametri base dell'ottimizzatore
        self.population_size = config.get("genetic.population_size", 100)
        self.generations = config.get("genetic.generations", 50)
        self.mutation_rate = config.get("genetic.mutation_rate", 0.1)
        self.elite_size = config.get("genetic.elite_size", 10)
        self.tournament_size = config.get("genetic.tournament_size", 5)
        self.min_trades = config.get("genetic.min_trades", 10)  # Added missing attribute
        
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
        
        # Pesi dell'ensemble
        self.ensemble_weights = {
            "base_gene": config.get("genetic.ensemble_weights.base_gene", 0.4),
            "volatility_gene": config.get("genetic.ensemble_weights.volatility_gene", 0.2),
            "momentum_gene": config.get("genetic.ensemble_weights.momentum_gene", 0.2),
            "pattern_gene": config.get("genetic.ensemble_weights.pattern_gene", 0.2)
        }
        
        # Abilitazione dei geni specializzati
        self.volatility_enabled = config.get("trading.volatility_gene.enabled", True)
        self.momentum_enabled = config.get("trading.momentum_gene.enabled", True)
        self.pattern_enabled = config.get("trading.pattern_gene.enabled", True)
        
        self.population = []
        self.best_ensemble = None
        self.generation_stats = []
        self.signal_combiner = EnsembleSignalCombiner(self.ensemble_weights)

        print(f"\nDispositivo utilizzato: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memoria GPU allocata: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")

    def evaluate_ensemble(self, ensemble: List[TorchGene], simulator: TradingSimulator) -> float:
        """Valuta un ensemble specifico"""
        try:
            simulator.run_simulation(ensemble[0])  # Per ora valutiamo solo il primo gene
            metrics = simulator.get_performance_metrics()
            
            print(f"         Metriche trading:")
            print(f"         - Trade totali: {metrics['total_trades']}")
            print(f"         - Win rate: {metrics['win_rate']*100:.1f}%")
            print(f"         - P&L: ${metrics['total_pnl']:.2f}")
            print(f"         - Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
            
            if metrics["total_trades"] < self.min_trades:  # Now using self.min_trades
                return 0
                
            # Calcola il fitness usando i pesi dalla configurazione
            profit_score = (
                min(1, metrics["total_pnl"] / simulator.initial_capital) * self.weights["profit_score"]["total_pnl"] +  
                (1 - metrics["max_drawdown"]) * self.weights["profit_score"]["max_drawdown"] +
                max(0, metrics["sharpe_ratio"]) * self.weights["profit_score"]["sharpe_ratio"]
            )
            
            quality_score = (
                metrics["win_rate"] * self.weights["quality_score"]["win_rate"] +
                min(1, metrics["total_trades"] / 100) * self.weights["quality_score"]["trade_frequency"]
            )
            
            fitness = (profit_score * self.weights["final"]["profit"] + 
                      quality_score * self.weights["final"]["quality"])
            
            return fitness

        except Exception as e:
            print(f"      ❌ Errore durante il calcolo del fitness")
            print(f"      Dettaglio errore: {str(e)}")
            raise
  
    def create_initial_population(self) -> List[List[TradingGene]]:
        """Crea la popolazione iniziale di ensemble di geni"""
        population = []
        
        for _ in range(self.population_size):
            ensemble = []
            
            # Gene base (sempre presente)
            ensemble.append(TradingGene(random_init=True))
            
            # Geni specializzati (se abilitati)
            if self.volatility_enabled:
                ensemble.append(VolatilityAdaptiveGene(random_init=True))
            if self.momentum_enabled:
                ensemble.append(MomentumGene(random_init=True))
            if self.pattern_enabled:
                ensemble.append(PatternRecognitionGene(random_init=True))
            
            population.append(ensemble)
        
        return population

    def create_next_generation(self, evaluated_population: List[Tuple[List[TradingGene], float]]):
        """Crea la prossima generazione (versione migliorata)"""
        sorted_population = sorted(evaluated_population, key=lambda x: x[1], reverse=True)
        new_population = []
        
        # Mantieni l'elite senza modifiche
        elite = [ensemble.copy() for ensemble, _ in sorted_population[:self.elite_size]]
        new_population.extend(elite)
        
        while len(new_population) < self.population_size:
            if np.random.random() < 0.8:  # 80% probabilità di crossover
                # Selezione dei genitori con tournament selection
                parent1 = self.select_parent(sorted_population)
                parent2 = self.select_parent(sorted_population)
                
                # Crossover tra gli ensemble
                child = self.crossover_ensembles(parent1, parent2)
                
                # Applica mutazione a ogni gene dell'ensemble con probabilità mutation_rate
                for gene in child:
                    if np.random.random() < self.mutation_rate:
                        gene.mutate(mutation_rate=np.random.uniform(0.2, 0.8))  # Mutation rate più aggressivo
            else:
                # 20% probabilità di creare un nuovo ensemble casuale
                child = self.create_random_ensemble()
            
            new_population.append(child)
        
        self.population = new_population

    def create_random_ensemble(self) -> List[TradingGene]:
        """Crea un nuovo ensemble completamente casuale"""
        ensemble = []
        
        # Gene base sempre presente con inizializzazione casuale
        ensemble.append(TradingGene(random_init=True))
        
        # Geni specializzati con inizializzazione casuale
        if self.volatility_enabled:
            ensemble.append(VolatilityAdaptiveGene(random_init=True))
        if self.momentum_enabled:
            ensemble.append(MomentumGene(random_init=True))
        if self.pattern_enabled:
            ensemble.append(PatternRecognitionGene(random_init=True))
        
        return ensemble

    def crossover_ensembles(self, parent1: List[TradingGene], parent2: List[TradingGene]) -> List[TradingGene]:
        """Esegue il crossover tra due ensemble (versione migliorata)"""
        child_ensemble = []
        
        # Assicuriamoci che i genitori abbiano lo stesso numero di geni
        assert len(parent1) == len(parent2), "Gli ensemble devono avere lo stesso numero di geni"
        
        for gene1, gene2 in zip(parent1, parent2):
            # Crossover uniforme del DNA tra i geni dello stesso tipo
            child_gene = gene1.__class__(random_init=False)  # Crea un nuovo gene dello stesso tipo
            child_dna = {}
            
            # Mischia casualmente i parametri del DNA dai genitori
            all_keys = set(gene1.dna.keys()) | set(gene2.dna.keys())
            for key in all_keys:
                if np.random.random() < 0.5:
                    child_dna[key] = gene1.dna.get(key, gene2.dna[key])
                else:
                    child_dna[key] = gene2.dna.get(key, gene1.dna[key])
                    
                # Aggiungi una piccola variazione casuale ai parametri numerici
                if isinstance(child_dna[key], (int, float)):
                    variation = np.random.uniform(-0.1, 0.1)
                    child_dna[key] *= (1 + variation)
            
            child_gene.dna = child_dna
            child_ensemble.append(child_gene)
        
        return child_ensemble

    def select_parent(self, evaluated_population: List[Tuple[List[TradingGene], float]]) -> List[TradingGene]:
        """Seleziona un genitore usando il tournament selection (versione migliorata)"""
        tournament_size = max(2, min(self.tournament_size, len(evaluated_population)))
        tournament = np.random.choice(len(evaluated_population), size=tournament_size, replace=False)
        tournament_pop = [evaluated_population[i] for i in tournament]
        
        # Selezione pesata basata sul fitness all'interno del torneo
        weights = [max(0.1, score) for _, score in tournament_pop]  # Evita pesi negativi
        weights = np.array(weights) / sum(weights)  # Normalizza i pesi
        
        selected_idx = np.random.choice(len(tournament_pop), p=weights)
        return tournament_pop[selected_idx][0]

    def evaluate_ensemble_parallel(self, args) -> Tuple[TorchGene, float]:
        """Funzione wrapper per la valutazione parallela degli ensemble"""
        try:
            ensemble_idx, ensemble, market_data_dict = args
            print(f"      Valutazione ensemble {ensemble_idx + 1}...")
            
            # Crea un nuovo simulatore per questo ensemble
            print(f"         Setup simulatore per ensemble {ensemble_idx + 1}")
            simulator = TradingSimulator()
            
            # Carica i dati di mercato
            print(f"         Caricamento dati per ensemble {ensemble_idx + 1}")
            for timeframe_str, data in market_data_dict.items():
                timeframe = TimeFrame(timeframe_str)
                df = pd.DataFrame(data)
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                simulator.add_market_data(timeframe, df)
            
            print(f"         Avvio simulazione ensemble {ensemble_idx + 1}")
            # Valuta l'ensemble
            fitness = self.evaluate_ensemble(ensemble, simulator)
            
            print(f"         Simulazione completata per ensemble {ensemble_idx + 1}")
            print(f"         Fitness: {fitness:.4f}")
            
            return ensemble, fitness
                
        except Exception as e:
            print(f"      ❌ Errore durante la valutazione dell'ensemble {ensemble_idx + 1}")
            print(f"      Dettaglio errore: {str(e)}")
            print(f"      Tipo errore: {type(e)}")
            import traceback
            print(f"      Stack trace:\n{traceback.format_exc()}")
            raise

    def calculate_fitness(self, metrics: Dict, initial_capital: float) -> float:
        """Calcola il fitness score considerando varie metriche"""
        if metrics["total_trades"] < config.get("genetic.min_trades", 10):
            return 0
            
        profit_weights = config.get("genetic.fitness_weights.profit_score", {})
        quality_weights = config.get("genetic.fitness_weights.quality_score", {})
        final_weights = config.get("genetic.fitness_weights.final_weights", {})
        
        # Calcola profit score
        profit_score = (
            min(1, metrics["total_pnl"] / initial_capital) * profit_weights.get("total_pnl", 0.4) +
            (1 - metrics["max_drawdown"]) * profit_weights.get("max_drawdown", 0.3) +
            max(0, metrics["sharpe_ratio"]) * profit_weights.get("sharpe_ratio", 0.3)
        )
        
        # Calcola quality score
        quality_score = (
            metrics["win_rate"] * quality_weights.get("win_rate", 0.6) +
            min(1, metrics["total_trades"] / 100) * quality_weights.get("trade_frequency", 0.4)
        )
        
        # Calcola score finale
        return (profit_score * final_weights.get("profit", 0.6) + 
                quality_score * final_weights.get("quality", 0.4))
  
    def optimize(self, simulator: TradingSimulator) -> Tuple[List[TorchGene], Dict]:
        print("\nInizio processo di ottimizzazione")
        print("1. Preparazione dati di mercato...")
        
        # Prepara i dati per la valutazione parallela
        market_data_dict = {}
        for timeframe, data in simulator.market_data.items():
            print(f"   Processando timeframe {timeframe.value}...")
            df_dict = pd.DataFrame([{
                'timestamp': d.timestamp,
                'open': d.open,
                'high': d.high,
                'low': d.low,
                'close': d.close,
                'volume': d.volume
            } for d in data])
            market_data_dict[timeframe.value] = df_dict
        
        print("2. Creazione popolazione iniziale...")
        self.population = self.create_initial_population()
        print(f"   Popolazione creata: {len(self.population)} individui")
        
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        print("3. Inizio ciclo generazioni...")
        for generation in range(self.generations):
            print(f"\nGenerazione {generation + 1}/{self.generations}")
            generation_start = datetime.now()
            
            try:
                print("   3.1 Preparazione valutazione...")
                eval_args = [(i, ensemble, market_data_dict) 
                            for i, ensemble in enumerate(self.population)]
                
                print("   3.2 Avvio valutazione parallela...")
                with Pool(processes=config.get("genetic.parallel_processes", 10)) as pool:
                    evaluated_population = list(pool.imap_unordered(
                        self.evaluate_ensemble_parallel, eval_args
                    ))
                print("   3.3 Valutazione completata")
                
                # Ordina per fitness
                evaluated_population.sort(key=lambda x: x[1], reverse=True)
                current_best = evaluated_population[0]
                
                print(f"   Best fitness questa generazione: {current_best[1]:.4f}")
                
                # Aggiorna il miglior ensemble
                if current_best[1] > best_fitness:
                    improvement = "N/A" if best_fitness == float('-inf') else \
                                f"{((current_best[1] - best_fitness) / abs(best_fitness) * 100):.1f}%"
                    print(f"   🎯 Nuovo miglior ensemble! Miglioramento: {improvement}")
                    best_fitness = current_best[1]
                    self.best_ensemble = current_best[0]
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                    print(f"   Generazioni senza miglioramenti: {generations_without_improvement}")
                
                # Aggiorna statistiche
                generation_time = (datetime.now() - generation_start).total_seconds()
                avg_fitness = np.mean([f for _, f in evaluated_population])
                
                self.generation_stats.append({
                    'generation': generation + 1,
                    'best_fitness': current_best[1],
                    'avg_fitness': avg_fitness,
                    'time': generation_time
                })
                
                print("   3.4 Creazione prossima generazione...")
                self.create_next_generation(evaluated_population)
                print("   3.5 Nuova generazione creata")
                
                if generations_without_improvement >= 10:
                    print("\n⚠️ Ottimizzazione terminata per mancanza di miglioramenti")
                    break
                    
            except Exception as e:
                print(f"Errore durante la generazione {generation + 1}: {str(e)}")
                raise
        
        return self.best_ensemble, {
            'best_fitness': best_fitness,
            'generation_stats': self.generation_stats
        }

    def print_ensemble_metrics(self, ensemble_idx: int, ensemble: List[TradingGene], 
                             metrics: Dict, elapsed_time: float):
        """Stampa le metriche dettagliate per un ensemble"""
        print(f"\n{'='*80}")
        print(f"VALUTAZIONE ENSEMBLE {ensemble_idx + 1}")
        print(f"{'='*80}")
        
        # Stampa composizione ensemble
        print("\nComposizione Ensemble:")
        for i, gene in enumerate(ensemble):
            gene_type = type(gene).__name__
            print(f"  {i+1}. {gene_type}")
        
        # Stampa metriche di performance
        print("\nMETRICHE DI PERFORMANCE:")
        print(f"  Trade totali: {metrics['total_trades']}")
        print(f"  Trade vincenti: {metrics['winning_trades']}")
        print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"  P&L: ${metrics['total_pnl']:.2f}")
        print(f"  Capitale finale: ${metrics['final_capital']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"\nTempo di valutazione: {elapsed_time:.2f} secondi")
        print(f"\n{'-'*80}")

