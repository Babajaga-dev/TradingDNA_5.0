import logging
import psutil
import torch
import intel_extension_for_pytorch as ipex

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
        self.gpu_backend = config.get("genetic.optimizer.gpu_backend", "auto")

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
                if device.type == "xpu":
                    # Monitoraggio Intel XPU
                    memory_allocated = torch.xpu.memory_allocated() / 1e9
                    memory_reserved = torch.xpu.memory_reserved() / 1e9
                    device_name = "Intel Arc GPU"
                elif device.type == "cuda":
                    # Monitoraggio NVIDIA CUDA
                    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
                    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                    device_name = torch.cuda.get_device_name(device.index)
                else:
                    continue  # Skip altri tipi di device
                
                utilization = memory_allocated / memory_reserved if memory_reserved > 0 else 0
                
                if (self.config["memory_warnings"] and 
                    utilization > self.config["threshold_utilization"]):
                    logger.warning(f"{device_name} memory utilization high: {utilization:.2%}")
                    
                logger.debug(f"{device_name} Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
                # Log metriche aggiuntive specifiche per backend
                if device.type == "xpu":
                    # Metriche specifiche XPU se disponibili
                    pass
                elif device.type == "cuda":
                    # Metriche specifiche CUDA
                    if hasattr(torch.cuda, 'memory_stats'):
                        stats = torch.cuda.memory_stats(device)
                        active_blocks = stats.get('active_blocks.all.current', 0)
                        logger.debug(f"CUDA Active Memory Blocks: {active_blocks}")
                
            except Exception as e:
                logger.error(f"Error monitoring device {device_name}: {str(e)}")

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
            
            # Monitora anche la memoria virtuale
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            logger.debug(f"Virtual Memory - Used: {vm.percent}%, Swap Used: {swap.percent}%")
            
        except Exception as e:
            logger.error(f"Error monitoring system resources: {str(e)}")

    def should_check_metrics(self, iteration: int) -> bool:
        """Determina se è il momento di controllare le metriche"""
        return (self.config["enabled"] and 
                iteration % (self.config["metrics_interval"]) == 0)

    def log_optimization_progress(self, stats: dict) -> None:
        """Logga i progressi dell'ottimizzazione"""
        try:
            logger.info("Optimization Progress:")
            logger.info(f"Best Fitness: {stats['best_fitness']:.4f}")
            logger.info(f"Average Fitness: {stats['avg_fitness']:.4f}")
            logger.info(f"Population Diversity: {stats['diversity']:.4f}")
            logger.info(f"Current Mutation Rate: {stats['mutation_rate']:.4f}")
            logger.info(f"Time: {stats['elapsed_time']:.2f}s")
            
            # Log metriche GPU se disponibili
            if self.use_gpu:
                if self.gpu_backend == "arc" and torch.xpu.is_available():
                    memory_allocated = torch.xpu.memory_allocated() / 1e9
                    logger.info(f"XPU Memory Usage: {memory_allocated:.2f}GB")
                elif torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    logger.info(f"CUDA Memory Usage: {memory_allocated:.2f}GB")
            
        except Exception as e:
            logger.error(f"Error logging optimization progress: {str(e)}")

    def get_memory_stats(self, device: torch.device) -> dict:
        """Ottiene statistiche dettagliate sulla memoria del dispositivo"""
        try:
            stats = {}
            
            if device.type == "xpu":
                stats.update({
                    'allocated': torch.xpu.memory_allocated() / 1e9,
                    'reserved': torch.xpu.memory_reserved() / 1e9,
                    'device_type': 'xpu'
                })
            elif device.type == "cuda":
                stats.update({
                    'allocated': torch.cuda.memory_allocated(device) / 1e9,
                    'reserved': torch.cuda.memory_reserved(device) / 1e9,
                    'device_type': 'cuda'
                })
                
                if hasattr(torch.cuda, 'memory_stats'):
                    cuda_stats = torch.cuda.memory_stats(device)
                    stats.update({
                        'active_blocks': cuda_stats.get('active_blocks.all.current', 0),
                        'inactive_split_blocks': cuda_stats.get('inactive_split_blocks.all.current', 0),
                        'allocated_bytes': cuda_stats.get('allocated_bytes.all.current', 0) / 1e9
                    })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {'error': str(e)}
