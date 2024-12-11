# src/optimization/genetic_stats.py
from dataclasses import dataclass

@dataclass
class OptimizationStats:
    """Statistiche di ottimizzazione"""
    best_fitness: float
    generations: int
    total_time: float
    avg_generation_time: float
    early_stopped: bool
    final_population_size: int
    final_diversity: float
    total_restarts: int
