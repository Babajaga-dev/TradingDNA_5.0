import logging
import psutil
import torch

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, config):
        self.config = {
            "enabled": config.get("genetic.performance_monitoring.enabled", True),
            "metrics_interval": config.get("genetic.performance_monitoring.metrics_interval", 2),
            "memory_warnings": config.get("genetic.performance_monitoring.memory_warnings", True),
            "threshold_utilization": config.get("genetic.performance_monitoring.threshold_utilization", 0.95)
        }
        self.use_gpu = config.get("genetic.optimizer.use_gpu", False)

    def check_performance(self, devices: list) -> None:
        """Monitora le performance e l'utilizzo delle risorse"""
        if not self.config["enabled"]:
            return
            
        try:
            if self.use_gpu:
                self._monitor_gpu(devices)
            self._monitor_system_resources()
            
        except Exception as e:
            logger.error(f"Error in performance monitoring: {str(e)}")

    def _monitor_gpu(self, devices: list) -> None:
        """Monitora l'utilizzo della GPU"""
        for i, device in enumerate(devices):
            try:
                memory_allocated = torch.cuda.memory_allocated(device) / 1e9
                memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                utilization = memory_allocated / memory_reserved if memory_reserved > 0 else 0
                
                if (self.config["memory_warnings"] and 
                    utilization > self.config["threshold_utilization"]):
                    logger.warning(f"GPU {i} memory utilization high: {utilization:.2%}")
                    
                logger.debug(f"GPU {i} Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
            except Exception as e:
                logger.error(f"Error monitoring GPU {i}: {str(e)}")

    def _monitor_system_resources(self) -> None:
        """Monitora l'utilizzo di CPU e RAM"""
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            ram_percent = process.memory_percent()
            
            if (self.config["memory_warnings"] and 
                ram_percent > self.config["threshold_utilization"] * 100):
                logger.warning(f"RAM utilization high: {ram_percent:.1f}%")
                
            logger.debug(f"CPU Usage: {cpu_percent:.1f}%, RAM Usage: {ram_percent:.1f}%")
            
        except Exception as e:
            logger.error(f"Error monitoring system resources: {str(e)}")

    def should_check_metrics(self, iteration: int) -> bool:
        """Determina se è il momento di controllare le metriche"""
        return (self.config["enabled"] and 
                iteration % (self.config["metrics_interval"]) == 0)

    def log_optimization_progress(self, stats: dict) -> None:
        """Logga i progressi dell'ottimizzazione"""
        try:
            logger.info(f"Best Fitness: {stats['best_fitness']:.4f}")
            logger.info(f"Average Fitness: {stats['avg_fitness']:.4f}")
            logger.info(f"Population Diversity: {stats['diversity']:.4f}")
            logger.info(f"Current Mutation Rate: {stats['mutation_rate']:.4f}")
            logger.info(f"Time: {stats['elapsed_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Error logging optimization progress: {str(e)}")
