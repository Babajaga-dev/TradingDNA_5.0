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
        # Parametri base
        self.population_size = config.get("genetic.population_size", 600)
        self.generations = config.get("genetic.generations", 200)
        self.mutation_rate = config.get("genetic.mutation_rate", 0.35)
        self.elite_size = config.get("genetic.elite_size", 2)
        self.tournament_size = config.get("genetic.tournament_size", 2)
        self.min_trades = config.get("genetic.min_trades", 50)
        self.num_processes = min(
            config.get("genetic.parallel_processes", 10),
            multiprocessing.cpu_count()
        )
        self.batch_size = config.get("genetic.batch_size", 32)
        
        # Nuovi parametri per restart e anti-convergenza
        self.mutation_decay = config.get("genetic.mutation_decay", 0.995)
        self.diversity_threshold = config.get("genetic.diversity_threshold", 0.15)
        self.restart_threshold = config.get("genetic.restart_threshold", 20)
        self.improvement_threshold = config.get("genetic.improvement_threshold", 0.001)
        self.restart_elite_fraction = config.get("genetic.restart_elite_fraction", 0.1)
        self.restart_mutation_multiplier = config.get("genetic.restart_mutation_multiplier", 2.0)
        
        # Stati
        self.generation_stats = []
        self.population = []
        self.precalculated_data = None
        self.best_gene = None
        self.best_fitness = float('-inf')
        
        logger.info("Initialized genetic optimizer with enhanced parameters")

    def _is_gene_different(self, gene: TradingGene, population: List[TradingGene], threshold: float = 0.2) -> bool:
        """Verifica se un gene è sufficientemente diverso dalla popolazione"""
        if not population:
            return True
            
        for existing_gene in population:
            differences = 0
            total_comparisons = 0
            
            for key in gene.dna:
                if key in existing_gene.dna:
                    if isinstance(gene.dna[key], (int, float)):
                        # For numeric values, calculate relative difference
                        rel_diff = abs(gene.dna[key] - existing_gene.dna[key]) / (abs(gene.dna[key]) + 1e-10)
                        differences += 1 if rel_diff > 0.1 else 0
                    elif isinstance(gene.dna[key], dict):
                        # For parameter dictionaries, compare each parameter
                        for param_key in gene.dna[key]:
                            if param_key in existing_gene.dna[key]:
                                param_val1 = gene.dna[key][param_key]
                                param_val2 = existing_gene.dna[key][param_key]
                                if isinstance(param_val1, (int, float)):
                                    rel_diff = abs(param_val1 - param_val2) / (abs(param_val1) + 1e-10)
                                    differences += 1 if rel_diff > 0.1 else 0
                                else:
                                    differences += 1 if param_val1 != param_val2 else 0
                                total_comparisons += 1
                    else:
                        # For other types, direct comparison
                        differences += 1 if gene.dna[key] != existing_gene.dna[key] else 0
                    total_comparisons += 1
            
            if total_comparisons > 0:
                difference_ratio = differences / total_comparisons
                if difference_ratio < threshold:
                    return False
                    
        return True

    def _check_for_restart(self) -> bool:
            """Verifica se è necessario un restart basato sui plateau"""
            if len(self.generation_stats) < self.restart_threshold:
                return False
                
            # Get last N generations fitness values
            recent_best_fitness = [stat['best_fitness'] 
                                for stat in self.generation_stats[-self.restart_threshold:]]
            recent_avg_fitness = [stat['avg_fitness']
                                for stat in self.generation_stats[-self.restart_threshold:]]
            
            # Calculate improvement metrics
            best_improvement = max(recent_best_fitness) - min(recent_best_fitness)
            avg_improvement = max(recent_avg_fitness) - min(recent_avg_fitness)
            avg_fitness_std = np.std(recent_avg_fitness)
            
            # Check multiple conditions for restart
            needs_restart = (
                best_improvement < self.improvement_threshold and  # No improvement in best fitness
                avg_improvement < self.improvement_threshold and   # No improvement in average fitness
                avg_fitness_std < self.improvement_threshold * 2   # Low variation in average fitness
            )
            
            if needs_restart:
                logger.info(f"Plateau detected for {self.restart_threshold} generations:")
                logger.info(f"Best improvement: {best_improvement:.6f}")
                logger.info(f"Average improvement: {avg_improvement:.6f}")
                logger.info(f"Average fitness std: {avg_fitness_std:.6f}")
            
            # Reset restart counter if we've seen significant improvement
            if best_improvement > self.improvement_threshold * 2:
                self.generations_without_improvement = 0
                
            return needs_restart

    def _perform_restart(self) -> None:
            """Esegue il restart della popolazione mantenendo alcune informazioni"""
            logger.info("Performing population restart...")
            
            # Calculate elite size and select elite
            elite_size = max(2, int(self.population_size * self.restart_elite_fraction))
            elite = sorted(self.population, 
                        key=lambda x: x.fitness_score or float('-inf'), 
                        reverse=True)[:elite_size]
            
            # Analyze elite features
            common_features = self._analyze_elite_features(elite)
            
            # Create new population
            new_population = elite.copy()  # Keep elite
            
            # Increase mutation rate temporarily
            temp_mutation_rate = min(0.8, self.mutation_rate * self.restart_mutation_multiplier)
            
            # Different strategies for creating new individuals
            while len(new_population) < self.population_size:
                strategy = np.random.choice(['elite_based', 'random', 'hybrid'], p=[0.4, 0.3, 0.3])
                
                if strategy == 'elite_based':
                    # Create new gene based on elite features
                    new_gene = self._create_gene_with_features(common_features)
                    new_gene.mutate(temp_mutation_rate)
                    
                elif strategy == 'random':
                    # Create completely random gene
                    new_gene = TradingGene(random_init=True)
                    
                else:  # hybrid
                    # Crossover between elite and random
                    parent1 = np.random.choice(elite)
                    parent2 = TradingGene(random_init=True)
                    new_gene = parent1.crossover(parent2)
                    new_gene.mutate(temp_mutation_rate)
                
                # Add to population if sufficiently different
                if self._is_gene_different(new_gene, new_population):
                    new_population.append(new_gene)
            
            self.population = new_population
            logger.info(f"Restart completed with {len(new_population)} individuals")

    def _analyze_elite_features(self, elite_genes: List[TradingGene]) -> Dict:
        """Analizza le caratteristiche comuni dei migliori geni"""
        features = {}
        
        for key in elite_genes[0].dna.keys():
            values = [gene.dna[key] for gene in elite_genes]
            
            if all(isinstance(v, (int, float)) for v in values):
                # Per valori numerici, calcola media e deviazione standard
                mean = np.mean(values)
                std = np.std(values)
                features[key] = {'mean': mean, 'std': std}
                
            elif all(isinstance(v, dict) for v in values):
                # Per dizionari, analizza ogni chiave separatamente
                param_features = {}
                param_keys = set().union(*(v.keys() for v in values))
                
                for param_key in param_keys:
                    param_values = [v.get(param_key) for v in values if param_key in v]
                    
                    if all(isinstance(pv, (int, float)) for pv in param_values):
                        param_features[param_key] = {
                            'mean': np.mean(param_values),
                            'std': np.std(param_values)
                        }
                    else:
                        # Per valori non numerici nei parametri, usa il più frequente
                        value_counts = {}
                        for pv in param_values:
                            if pv is not None:
                                value_counts[str(pv)] = value_counts.get(str(pv), 0) + 1
                        most_common = max(value_counts.items(), key=lambda x: x[1])[0]
                        param_features[param_key] = {'value': most_common}
                        
                features[key] = {'params': param_features}
                
            else:
                # Per altri tipi (es. stringhe), usa conteggio valori
                value_counts = {}
                for v in values:
                    str_v = str(v)  # Converti in stringa per garantire hashability
                    value_counts[str_v] = value_counts.get(str_v, 0) + 1
                most_common = max(value_counts.items(), key=lambda x: x[1])[0]
                features[key] = {'value': most_common}
        
        return features

    def _create_gene_with_features(self, features: Dict) -> TradingGene:
        """Crea un nuovo gene basato sulle caratteristiche comuni"""
        new_gene = TradingGene(random_init=False)
        
        for key, feature in features.items():
            if 'mean' in feature:
                # Per valori numerici
                std = max(feature['std'], feature['mean'] * 0.1)  # Evita std troppo piccoli
                value = np.random.normal(feature['mean'], std)
                
                if isinstance(feature['mean'], int):
                    value = int(round(value))
                    
                new_gene.dna[key] = value
                
            elif 'params' in feature:
                # Per dizionari di parametri
                new_gene.dna[key] = {}
                for param_key, param_feature in feature['params'].items():
                    if 'mean' in param_feature:
                        # Parametro numerico
                        std = max(param_feature['std'], param_feature['mean'] * 0.1)
                        value = np.random.normal(param_feature['mean'], std)
                        
                        if isinstance(param_feature['mean'], int):
                            value = int(round(value))
                        
                        new_gene.dna[key][param_key] = value
                    else:
                        # Parametro non numerico
                        new_gene.dna[key][param_key] = param_feature['value']
                        
            else:
                # Per altri tipi (es. stringhe)
                new_gene.dna[key] = feature['value']
        
        return new_gene

    def _calculate_diversity(self) -> float:
        """Calcola la diversità della popolazione"""
        try:
            if not self.population:
                return 0.0
                
            unique_signatures = set()
            total_genes = len(self.population)
            
            for gene in self.population:
                # Normalizza i valori numerici per considerare piccole variazioni
                signature = []
                for key, value in sorted(gene.dna.items()):
                    if isinstance(value, dict):
                        # Gestisci dizionari innestati
                        dict_items = []
                        for k, v in sorted(value.items()):
                            if isinstance(v, (int, float)):
                                # Arrotonda i valori numerici per considerare simili quelli vicini
                                v = round(v, 2)
                            dict_items.append((k, v))
                        signature.append((key, tuple(dict_items)))
                    elif isinstance(value, (int, float)):
                        # Arrotonda i valori numerici
                        signature.append((key, round(value, 2)))
                    else:
                        signature.append((key, value))
                
                unique_signatures.add(tuple(signature))
            
            diversity = len(unique_signatures) / total_genes
            logger.debug(f"Population diversity: {diversity:.4f} ({len(unique_signatures)} unique in {total_genes} total)")
            return diversity
            
        except Exception as e:
            logger.error(f"Error calculating diversity: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def _selection_and_reproduction(self, 
                                  population: List[TradingGene], 
                                  fitness_scores: List[float]) -> List[TradingGene]:
        """Esegue selezione e riproduzione con meccanismi anti-stagnazione"""
        try:
            # Normalizza i fitness scores
            fitness_scores = np.array(fitness_scores)
            fitness_scores = (fitness_scores - fitness_scores.min()) / (fitness_scores.max() - fitness_scores.min() + 1e-10)
            
            # Seleziona elite con controllo duplicati
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            new_population = []
            elite_signatures = set()
            
            for idx in elite_indices:
                gene = deepcopy(population[idx])
                signature = self._get_gene_signature(gene)
                if signature not in elite_signatures:
                    elite_signatures.add(signature)
                    new_population.append(gene)
            
            # Calcola probabilità di selezione con temperature scaling
            temperature = 0.1  # Controlla la pressione selettiva
            selection_probs = np.exp(fitness_scores / temperature)
            selection_probs = selection_probs / selection_probs.sum()
            
            # Aggiorna mutation_rate con decay e floor
            min_mutation_rate = 0.05  # Tasso minimo di mutazione
            current_generation = len(self.generation_stats)
            current_mutation_rate = max(
                min_mutation_rate,
                self.mutation_rate * (self.mutation_decay ** current_generation)
            )
            
            # Reproduction loop
            while len(new_population) < self.population_size:
                if np.random.random() < 0.8:  # 80% crossover, 20% mutation
                    # Seleziona genitori diversi
                    parent1 = self._tournament_selection(population, selection_probs)
                    parent2 = self._tournament_selection(population, selection_probs)
                    while self._get_gene_signature(parent1) == self._get_gene_signature(parent2):
                        parent2 = self._tournament_selection(population, selection_probs)
                    
                    # Crossover con mutation
                    child = parent1.crossover(parent2)
                    child.mutate(current_mutation_rate)
                else:
                    # Mutation forte
                    parent = self._tournament_selection(population, selection_probs)
                    child = deepcopy(parent)
                    child.mutate(current_mutation_rate * 2)
                
                # Verifica unicità
                child_signature = self._get_gene_signature(child)
                if child_signature not in elite_signatures:
                    new_population.append(child)
                    elite_signatures.add(child_signature)
            
            # Aggiungi variazione casuale se necessario
            if len(new_population) < self.population_size:
                missing = self.population_size - len(new_population)
                for _ in range(missing):
                    new_gene = TradingGene(random_init=True)
                    new_population.append(new_gene)
            
            return new_population
            
        except Exception as e:
            logger.error(f"Error in selection and reproduction: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _get_gene_signature(self, gene: TradingGene) -> tuple:
        """Genera una signature unica per un gene"""
        signature = []
        for key, value in sorted(gene.dna.items()):
            if isinstance(value, dict):
                dict_items = []
                for k, v in sorted(value.items()):
                    if isinstance(v, (int, float)):
                        v = round(v, 2)
                    dict_items.append((k, v))
                signature.append((key, tuple(dict_items)))
            elif isinstance(value, (int, float)):
                signature.append((key, round(value, 2)))
            else:
                signature.append((key, value))
        return tuple(signature)

    def _inject_diversity(self, population: List[TradingGene]) -> None:
        """Inietta diversità nella popolazione quando necessario"""
        num_random = int(self.population_size * 0.2)  # 20% nuovi individui casuali
        population[-num_random:] = [TradingGene(random_init=True) 
                                  for _ in range(num_random)]

    def optimize(self, simulator: TradingSimulator) -> Tuple[TradingGene, Dict]:
        """Esegue l'ottimizzazione genetica con restart e anti-convergenza"""
        try:
            logger.info("Starting enhanced genetic optimization")
            start_time = datetime.now()
            
            self.precalculated_data = self._precalculate_indicators(simulator.market_state)
            self.population = self._initialize_population()
            
            generations_without_improvement = 0
            best_overall_fitness = float('-inf')
            
            for generation in range(self.generations):
                generation_start = datetime.now()
                logger.info(f"Generation {generation + 1}/{self.generations}")
                
                # Valuta popolazione
                fitness_scores, best_gen_fitness, best_gen_gene = self._evaluate_population(simulator)
                
                # Aggiorna migliori risultati
                if best_gen_fitness > best_overall_fitness:
                    best_overall_fitness = best_gen_fitness
                    self.best_gene = deepcopy(best_gen_gene)
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Calcola statistiche
                diversity = self._calculate_diversity()
                gen_stats = {
                    "generation": generation + 1,
                    "best_fitness": best_gen_fitness,
                    "avg_fitness": np.mean(fitness_scores),
                    "std_fitness": np.std(fitness_scores),
                    "diversity": diversity,
                    "elapsed_time": (datetime.now() - generation_start).total_seconds()
                }
                self.generation_stats.append(gen_stats)
                
                # Log dei progressi
                logger.info(f"Best Fitness: {gen_stats['best_fitness']:.4f}")
                logger.info(f"Average Fitness: {gen_stats['avg_fitness']:.4f}")
                logger.info(f"Population Diversity: {gen_stats['diversity']:.4f}")
                logger.info(f"Time: {gen_stats['elapsed_time']:.2f}s")
                
                # Verifica necessità di restart
                if self._check_for_restart():
                    logger.info("Performing population restart...")
                    self._perform_restart()
                    generations_without_improvement = 0
                
                # Early stopping con controllo diversità
                if generations_without_improvement >= self.restart_threshold and \
                   diversity < self.diversity_threshold:
                    logger.info("Early stopping - No improvement and low diversity")
                    break
                
                # Crea prossima generazione
                if generation < self.generations - 1:
                    self.population = self._selection_and_reproduction(
                        self.population, fitness_scores
                    )
            
            # Statistiche finali
            total_time = (datetime.now() - start_time).total_seconds()
            final_stats = {
                "best_fitness": best_overall_fitness,
                "generations": len(self.generation_stats),
                "total_time": total_time,
                "avg_generation_time": total_time / len(self.generation_stats),
                "early_stopped": generations_without_improvement >= self.restart_threshold,
                "final_population_size": len(self.population),
                "final_diversity": self._calculate_diversity()
            }
            
            logger.info("\nOptimization completed!")
            logger.info(f"Best fitness achieved: {best_overall_fitness:.4f}")
            logger.info(f"Total generations: {final_stats['generations']}")
            logger.info(f"Total time: {final_stats['total_time']:.2f}s")
            
            return self.best_gene, final_stats
            
        except Exception as e:
            logger.error("Error in genetic optimization:")
            logger.error(str(e))
            logger.error(traceback.format_exc())
            raise

    def _calculate_fitness(self, metrics: Dict) -> float:
        """Calcola il fitness score con nuovi pesi e metriche"""
        if metrics["total_trades"] < self.min_trades:
            return 0.0
            
        try:
            weights = config.get("genetic.fitness_weights", {})
            profit_weights = weights.get("profit_score", {})
            quality_weights = weights.get("quality_score", {})
            final_weights = weights.get("final_weights", {})
            
            # Componente profit
            profit_score = (
                profit_weights.get("total_pnl", 0.30) * metrics["total_pnl"] / 10000 +
                profit_weights.get("max_drawdown", 0.35) * (1 - metrics["max_drawdown"]) +
                profit_weights.get("sharpe_ratio", 0.35) * max(0, metrics["sharpe_ratio"]) / 3
            )
            
            # Componente quality
            quality_score = (
                quality_weights.get("win_rate", 0.4) * metrics["win_rate"] +
                quality_weights.get("trade_frequency", 0.4) * min(1.0, metrics["total_trades"] / 100)
            )
            
            # Nuova metrica di consistenza
            if "profit_factor" in metrics:
                consistency_score = quality_weights.get("consistency", 0.2) * \
                                  (metrics["profit_factor"] - 1) / 2
                quality_score += consistency_score
            
            # Calcola diversità se richiesta
            diversity_score = 0.0
            if final_weights.get("diversity", 0) > 0:
                diversity_score = self._calculate_diversity()
            
            # Score finale con pesi aggiornati
            final_score = (
                final_weights.get("profit", 0.45) * profit_score +
                final_weights.get("quality", 0.45) * quality_score +
                final_weights.get("diversity", 0.1) * diversity_score
            )
            
            # Penalità
            penalties = 1.0
            if metrics["total_trades"] > 500:
                penalties *= 0.8
            if metrics["max_drawdown"] > 0.3:
                penalties *= 0.7
            if metrics["win_rate"] < 0.4:
                penalties *= 0.9
            
            return max(0.0, final_score * penalties)
            
        except Exception as e:
            logger.error(f"Error calculating fitness: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

# Aggiorna questa parte nel file src/optimization/genetic.py

    def _precalculate_indicators(self, market_state: MarketState) -> Dict[str, np.ndarray]:
        """Precalcola indicatori tecnici comuni con periodi estesi"""
        logger.info("Precalculating technical indicators...")
        
        indicators = {}
        try:
            periods = TradingGene.VALID_PERIODS
            logger.info(f"Calculating indicators for {len(periods)} periods: {periods}")
            
            # Raggruppa i calcoli per tipo di indicatore per ottimizzare le performance
            for period in periods:
                # Moving Averages
                sma = talib.SMA(market_state.close, timeperiod=period)
                indicators[f"SMA_{period}"] = sma
                
                ema = talib.EMA(market_state.close, timeperiod=period)
                indicators[f"EMA_{period}"] = ema
                
                # Oscillators
                rsi = talib.RSI(market_state.close, timeperiod=period)
                indicators[f"RSI_{period}"] = rsi
                
                # Volatility
                upper, middle, lower = talib.BBANDS(
                    market_state.close, 
                    timeperiod=period,
                    nbdevup=2,
                    nbdevdn=2,
                    matype=talib.MA_Type.SMA
                )
                indicators[f"BB_UPPER_{period}"] = upper
                indicators[f"BB_LOWER_{period}"] = lower
                indicators[f"BB_MIDDLE_{period}"] = middle

                # Average True Range per volatilità
                atr = talib.ATR(
                    market_state.high,
                    market_state.low,
                    market_state.close,
                    timeperiod=period
                )
                indicators[f"ATR_{period}"] = atr

            # Aggiungi dati grezzi
            indicators["CLOSE"] = market_state.close
            indicators["HIGH"] = market_state.high
            indicators["LOW"] = market_state.low

            # Aggiungi indicatori compositi per periodi selezionati
            for period in TradingGene.PERIOD_GROUPS['medium_term']:
                # MACD con periodi proporzionali
                fast_period = max(2, int(period/3))
                slow_period = period
                signal_period = max(2, int(period/4))
                
                macd, signal, hist = talib.MACD(
                    market_state.close,
                    fastperiod=fast_period,
                    slowperiod=slow_period,
                    signalperiod=signal_period
                )
                indicators[f"MACD_{period}"] = macd
                indicators[f"MACD_SIGNAL_{period}"] = signal
                indicators[f"MACD_HIST_{period}"] = hist

                # Stochastic RSI
                fastk_period = max(2, int(period/4))
                fastd_period = max(2, int(period/8))
                
                fastk, fastd = talib.STOCHRSI(
                    market_state.close,
                    timeperiod=period,
                    fastk_period=fastk_period,
                    fastd_period=fastd_period
                )
                indicators[f"STOCHRSI_K_{period}"] = fastk
                indicators[f"STOCHRSI_D_{period}"] = fastd

                # ADX per trend strength
                adx = talib.ADX(
                    market_state.high,
                    market_state.low,
                    market_state.close,
                    timeperiod=period
                )
                indicators[f"ADX_{period}"] = adx

            # Cache avanzata per pattern recognition
            for period in TradingGene.PERIOD_GROUPS['short_term']:
                # Momentum
                mom = talib.MOM(market_state.close, timeperiod=period)
                indicators[f"MOM_{period}"] = mom
                
                # Rate of Change
                roc = talib.ROC(market_state.close, timeperiod=period)
                indicators[f"ROC_{period}"] = roc
                
                # Williams %R
                willr = talib.WILLR(
                    market_state.high,
                    market_state.low,
                    market_state.close,
                    timeperiod=period
                )
                indicators[f"WILLR_{period}"] = willr

            # Candlestick pattern recognition
            indicators["CDL_DOJI"] = talib.CDLDOJI(
                market_state.open,
                market_state.high,
                market_state.low,
                market_state.close
            )
            indicators["CDL_ENGULFING"] = talib.CDLENGULFING(
                market_state.open,
                market_state.high,
                market_state.low,
                market_state.close
            )
            indicators["CDL_HAMMER"] = talib.CDLHAMMER(
                market_state.open,
                market_state.high,
                market_state.low,
                market_state.close
            )

            logger.info(f"Precalculated {len(indicators)} indicators")
            logger.debug(f"Available indicators: {sorted(indicators.keys())}")

            return indicators
            
        except Exception as e:
            logger.error(f"Error precalculating indicators: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _initialize_population(self) -> List[TradingGene]:
        """Inizializza popolazione con diversità garantita"""
        logger.info("Initializing genetic population...")
        population = []
        max_attempts = 3  # Numero massimo di tentativi di rigenerazione
        
        for attempt in range(max_attempts):
            population.clear()
            unique_signatures = set()
            
            # Crea geni assicurando diversità minima
            while len(population) < self.population_size:
                gene = TradingGene(random_init=True)
                
                # Crea signature del gene
                signature = []
                for key, value in sorted(gene.dna.items()):
                    if isinstance(value, dict):
                        # Converti dict in tuple ordinate
                        dict_items = sorted(value.items())
                        signature.append((key, tuple(dict_items)))
                    else:
                        signature.append((key, value))
                
                signature = tuple(signature)
                
                # Aggiungi solo se la signature è unica
                if signature not in unique_signatures:
                    unique_signatures.add(signature)
                    population.append(gene)
                    
                    # Ogni 50 geni, verifica la diversità
                    if len(population) % 50 == 0:
                        current_diversity = len(unique_signatures) / self.population_size
                        logger.debug(f"Current population: {len(population)}, Diversity: {current_diversity:.4f}")
            
            final_diversity = len(unique_signatures) / self.population_size
            logger.info(f"Population initialized with diversity: {final_diversity:.4f}")
            
            if final_diversity >= self.diversity_threshold:
                break
            elif attempt < max_attempts - 1:
                logger.info(f"Diversity too low ({final_diversity:.4f}), attempt {attempt + 2}/{max_attempts}")
            
        logger.info(f"Successfully initialized {len(population)} genes")
        return population

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

    def _tournament_selection(self, population: List[TradingGene], selection_probs: np.ndarray) -> TradingGene:
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