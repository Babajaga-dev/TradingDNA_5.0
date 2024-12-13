import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from datetime import datetime
from multiprocessing import Pool, cpu_count

from ..models.genes import TorchGene
from ..models.common import TimeFrame
from ..models.simulator import TradingSimulator

logger = logging.getLogger(__name__)

class EnsemblePopulationManager:
    """Gestisce la popolazione di ensemble e la sua valutazione"""
    
    def __init__(self, batch_size: int, evaluator):
        self.batch_size = batch_size
        self.evaluator = evaluator
        self.generation_stats: List[Dict[str, Any]] = []

    def evaluate_population_parallel(self, 
                                  population: List[List[TorchGene]], 
                                  market_data_dict: Dict[str, Any],
                                  signal_combiner) -> List[Tuple[List[TorchGene], float]]:
        """
        Valuta la popolazione in parallelo
        
        Args:
            population: Lista di ensemble da valutare
            market_data_dict: Dati di mercato
            signal_combiner: Combinatore di segnali
            
        Returns:
            Lista di tuple (ensemble, fitness)
        """
        try:
            num_workers = min(cpu_count(), 8)  # Limita a max 8 workers
            eval_args = [(i, ensemble, market_data_dict, signal_combiner) 
                        for i, ensemble in enumerate(population)]
            
            evaluated_population: List[Tuple[List[TorchGene], float]] = []
            total_batches = (len(eval_args) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(total_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, len(eval_args))
                current_batch = eval_args[batch_start:batch_end]
                
                with Pool(num_workers) as pool:
                    batch_results = list(pool.imap_unordered(
                        self._evaluate_ensemble_parallel, 
                        current_batch
                    ))
                    evaluated_population.extend(batch_results)
            
            return sorted(evaluated_population, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in parallel evaluation: {e}")
            return [(ensemble, 0.0) for ensemble in population]

    def _evaluate_ensemble_parallel(self, args: Tuple[int, List[TorchGene], Dict[str, Any], Any]) -> Tuple[List[TorchGene], float]:
        """
        Valutazione parallela di un ensemble
        
        Args:
            args: Tupla (indice, ensemble, dati di mercato, signal_combiner)
            
        Returns:
            Tupla (ensemble, fitness)
        """
        try:
            ensemble_idx, ensemble, market_data_dict, signal_combiner = args
            
            simulator = TradingSimulator()
            
            for timeframe_str, data in market_data_dict.items():
                timeframe = TimeFrame(timeframe_str)
                df = pd.DataFrame(data)
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                simulator.add_market_data(timeframe, df)
                
            return ensemble, self.evaluator.evaluate_ensemble(ensemble, simulator, signal_combiner)
            
        except Exception as e:
            logger.error(f"Error in parallel evaluation: {e}")
            return ensemble, 0.0

    def prepare_market_data(self, simulator: TradingSimulator) -> Dict[str, Any]:
        """
        Prepara i dati di mercato per la parallelizzazione
        
        Args:
            simulator: Simulatore di trading
            
        Returns:
            Dati di mercato formattati
        """
        try:
            market_data_dict = {}
            for timeframe, data in simulator.market_data.items():
                df_dict = pd.DataFrame([{
                    'timestamp': d.timestamp,
                    'open': d.open,
                    'high': d.high,
                    'low': d.low,
                    'close': d.close,
                    'volume': d.volume
                } for d in data])
                market_data_dict[timeframe.value] = df_dict.to_dict('records')
            return market_data_dict
        except Exception as e:
            logger.error(f"Error preparing market data: {e}")
            return {}

    def update_generation_stats(self, 
                              generation: int,
                              evaluated_population: List[Tuple[List[TorchGene], float]],
                              generation_start: datetime) -> None:
        """
        Aggiorna le statistiche della generazione
        
        Args:
            generation: Numero della generazione
            evaluated_population: Popolazione valutata
            generation_start: Timestamp inizio generazione
        """
        try:
            generation_time = (datetime.now() - generation_start).total_seconds()
            avg_fitness = np.mean([f for _, f in evaluated_population])
            
            self.generation_stats.append({
                'generation': generation + 1,
                'best_fitness': evaluated_population[0][1],
                'avg_fitness': avg_fitness,
                'time': generation_time
            })
        except Exception as e:
            logger.error(f"Error updating generation stats: {e}")

    def get_generation_stats(self) -> List[Dict[str, Any]]:
        """
        Restituisce le statistiche delle generazioni
        
        Returns:
            Lista di statistiche per generazione
        """
        return self.generation_stats
