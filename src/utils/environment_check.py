# src/utils/environment_check.py
import sys
import torch
import psutil
import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class EnvironmentChecker:
    def __init__(self):
        self.checks_passed: List[str] = []
        self.checks_failed: List[str] = []
        self.warnings: List[str] = []

    def check_cuda_installation(self) -> bool:
        """Verifica l'installazione CUDA"""
        try:
            if not torch.cuda.is_available():
                self.checks_failed.append("CUDA non disponibile")
                return False

            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            
            self.checks_passed.append(f"CUDA version: {cuda_version}")
            self.checks_passed.append(f"GPU devices found: {device_count}")
            
            # Verifica ogni GPU
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                mem_free, mem_total = torch.cuda.mem_get_info(i)
                self.checks_passed.append(
                    f"GPU {i}: {props.name} - "
                    f"Compute: {props.major}.{props.minor} - "
                    f"Memory: {mem_total/1024**3:.1f}GB"
                )
            
            return True
            
        except Exception as e:
            self.checks_failed.append(f"Errore verifica CUDA: {str(e)}")
            return False

    def check_linux_configuration(self) -> bool:
        """Verifica la configurazione Linux"""
        if platform.system() != 'Linux':
            return True

        try:
            # Verifica permessi e file system
            checks: Dict[str, bool] = {
                'hugepages': self._check_hugepages(),
                'tmp_access': self._check_tmp_access(),
                'gpu_permissions': self._check_gpu_permissions()
            }

            if all(checks.values()):
                self.checks_passed.append("Linux configuration: OK")
                return True
            else:
                failed = [k for k, v in checks.items() if not v]
                self.checks_failed.extend(f"Linux {f} check failed" for f in failed)
                return False

        except Exception as e:
            self.checks_failed.append(f"Linux configuration error: {str(e)}")
            return False

    def _check_hugepages(self) -> bool:
        """Verifica configurazione hugepages"""
        try:
            with open('/proc/sys/vm/nr_hugepages', 'r') as f:
                hugepages = int(f.read().strip())
                if hugepages < 128:
                    self.warnings.append(
                        "Hugepages < 128, performance might be affected"
                    )
                return True
        except:
            self.warnings.append("Could not check hugepages configuration")
            return False

    def _check_tmp_access(self) -> bool:
        """Verifica accesso /tmp"""
        tmp_path = Path('/tmp')
        try:
            test_file = tmp_path / f'test_{id(self)}'
            test_file.touch()
            test_file.unlink()
            return True
        except:
            return False

    def _check_gpu_permissions(self) -> bool:
        """Verifica permessi GPU"""
        try:
            result = subprocess.run(
                ['nvidia-smi'], 
                capture_output=True, 
                text=True,
                timeout=10  # Aggiunto timeout di 10 secondi
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False

    def check_memory_configuration(self) -> bool:
        """Verifica configurazione memoria"""
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        available_gb = vm.available / (1024**3)
        
        if total_gb < 8:
            self.checks_failed.append(f"Insufficient RAM: {total_gb:.1f}GB")
            return False
            
        if available_gb < 4:
            self.warnings.append(f"Low available memory: {available_gb:.1f}GB")
            
        self.checks_passed.append(
            f"Memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available"
        )
        return True

    def check_pytorch_configuration(self) -> bool:
        """Verifica configurazione PyTorch"""
        try:
            # Verifica versione PyTorch
            version = torch.__version__
            build = torch.__config__.show()
            
            if torch.cuda.is_available():
                # Test basic CUDA operations
                x = torch.randn(100, 100).cuda()
                y = torch.matmul(x, x)
                del x, y
                torch.cuda.empty_cache()
                
                self.checks_passed.append(
                    f"PyTorch {version} with CUDA support: OK"
                )
            else:
                self.warnings.append(
                    f"PyTorch {version} CPU only"
                )
            
            return True
            
        except Exception as e:
            self.checks_failed.append(f"PyTorch check failed: {str(e)}")
            return False

    def run_all_checks(self) -> bool:
        """Esegue tutti i controlli"""
        checks = [
            self.check_cuda_installation,
            self.check_linux_configuration,
            self.check_memory_configuration,
            self.check_pytorch_configuration
        ]
        
        results = []
        for check in checks:
            try:
                results.append(check())
            except Exception as e:
                self.checks_failed.append(f"Check failed: {str(e)}")
                results.append(False)
        
        return all(results)

    def print_report(self) -> None:
        """Stampa report dei controlli"""
        print("\nEnvironment Check Report")
        print("="*50)
        
        if self.checks_passed:
            print("\nPassed Checks:")
            for check in self.checks_passed:
                print(f"✓ {check}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"! {warning}")
        
        if self.checks_failed:
            print("\nFailed Checks:")
            for check in self.checks_failed:
                print(f"✗ {check}")
        
        print("\nSummary:")
        total = len(self.checks_passed) + len(self.checks_failed)
        print(f"Total checks: {total}")
        print(f"Passed: {len(self.checks_passed)}")
        print(f"Failed: {len(self.checks_failed)}")
        print(f"Warnings: {len(self.warnings)}")

def main() -> int:
    """Entry point per verifica ambiente"""
    logging.basicConfig(level=logging.INFO)
    
    checker = EnvironmentChecker()
    all_passed = checker.run_all_checks()
    checker.print_report()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
