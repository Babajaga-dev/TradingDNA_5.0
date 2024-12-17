import torch
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from ..models.genes import TorchGene, create_ensemble_gene
from ..models.simulator import TradingSimulator
from ..utils.config import config
from .ensemble_signals import EnsembleSignalCombiner
from .ensemble_metrics import EnsembleEvaluator
from .ensemble_genetic import EnsembleGeneticOperator
from .ensemble_population import EnsemblePopulationManager

logger = logging.getLogger(__name__)

class EnsembleGeneOptimizer:
    """Ottimizzatore per ensemble di geni"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: Device PyTorch opzionale
        """
        # Setup device
        if device is not None:
            self.device = device
        else:
            gpu_backend = config.get("genetic.optimizer.gpu_backend", "auto")
            use_gpu = config.get("genetic.optimizer.use_gpu", False)
            
            if not use_gpu:
                self.device = torch.device("cpu")
            elif gpu_backend == "arc":
                try:
                    import intel_extension_for_pytorch as ipex
                    if torch.xpu.is_available():
                        self.device = torch.device("xpu")
                        # Ottimizza per XPU
                        ipex.optimize()
                    else:
                        self.device = torch.device("cpu")
                except ImportError:
                    logger.warning("Intel Extension for PyTorch not found, using CPU")
                    self.device = torch.device("cpu")
            elif (gpu_backend == "cuda" or gpu_backend == "auto") and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        
        # Configura mixed precision
        self.mixed_precision = config.get("genetic.optimizer.device_config.mixed_precision", False)
        if self.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler()
        
        # Parametri base
        self.population_size = config.get("genetic.population_size", 100)
        self.generations = config.get("genetic.generations", 50)
        self.mutation_rate = config.get("genetic.mutation_rate", 0.1)
        self.elite_size = config.get("genetic.elite_size", 10)
        self.tournament_size = config.get("genetic.tournament_size", 5)
        self.batch_size = config.get("genetic.batch_size", 32)
        self.min_trades = config.get("genetic.min_trades", 10)
        
        # Pesi ensemble
        self.ensemble_weights = {
            "base_gene": config.get("genetic.ensemble_weights.base_gene", 0.4),
            "volatility_gene": config.get("genetic.ensemble_weights.volatility_gene", 0.2),
            "momentum_gene": config.get("genetic.ensemble_weights.momentum_gene", 0.2),
            "pattern_gene": config.get("genetic.ensemble_weights.pattern_gene", 0.2)
        }
        
        # Inizializza componenti
        self.signal_combiner = EnsembleSignalCombiner(self.ensemble_weights)
        self.evaluator = EnsembleEvaluator(self.min_trades)
        self.genetic_operator = EnsembleGeneticOperator(
            mutation_rate=self.mutation_rate,
            tournament_size=self.tournament_size
        )
        self.population_manager = EnsemblePopulationManager(
            batch_size=self.batch_size,
            evaluator=self.evaluator
        )
        
        # Stati
        self.population: List[List[TorchGene]] = []
        self.best_ensemble: Optional[List[TorchGene]] = None
        
        # Log configurazione
        logger.info(f"Initialized EnsembleGeneOptimizer with device: {self.device}")
        if self.mixed_precision:
            logger.info("Mixed precision enabled")

    def optimize(self, simulator: TradingSimulator) -> Tuple[List[TorchGene], Dict[str, Any]]:
        """
        Esegue ottimizzazione dell'ensemble
        
        Args:
            simulator: Simulatore di trading
            
        Returns:
            Tupla (miglior ensemble, statistiche)
        """
        try:
            # Inizializza popolazione
            self.population = self.genetic_operator.create_initial_population(self.population_size)
            best_fitness = float('-inf')
            generations_without_improvement = 0
            
            # Prepara dati per parallelizzazione
            market_data_dict = self.population_manager.prepare_market_data(simulator)

            # Ciclo principale di evoluzione
            for generation in range(self.generations):
                generation_start = datetime.now()
                
                # Gestione memoria GPU
                if self.device.type != "cpu":
                    if self.device.type == "xpu":
                        torch.xpu.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                
                # Valuta popolazione
                evaluated_population = self.population_manager.evaluate_population_parallel(
                    self.population,
                    market_data_dict,
                    self.signal_combiner
                )
                
                # Aggiorna migliore
                current_best = evaluated_population[0]
                if current_best[1] > best_fitness:
                    best_fitness = current_best[1]
                    self.best_ensemble = current_best[0]
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Aggiorna statistiche
                self.population_manager.update_generation_stats(
                    generation,
                    evaluated_population,
                    generation_start
                )
                
                # Verifica condizione di stop
                if generations_without_improvement >= 10:
                    break
                    
                # Crea nuova generazione
                self.population = self.genetic_operator.create_next_generation(
                    evaluated_population,
                    self.population_size,
                    self.elite_size
                )
            
            # Cleanup finale
            if self.device.type != "cpu":
                if self.device.type == "xpu":
                    torch.xpu.empty_cache()
                else:
                    torch.cuda.empty_cache()
            
            # Gestisci caso di fallimento
            if self.best_ensemble is None:
                self.best_ensemble = create_ensemble_gene(random_init=True)
                best_fitness = 0.0
            
            return self.best_ensemble, {
                'best_fitness': best_fitness,
                'generation_stats': self.population_manager.get_generation_stats()
            }
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return create_ensemble_gene(random_init=True), {
                'error': str(e),
                'best_fitness': 0.0,
                'generation_stats': []
            }
