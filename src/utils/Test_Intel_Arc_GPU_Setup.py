import torch; 
import intel_extension_for_pytorch as ipex;

import time

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

# Operazione su CPU
a_cpu = torch.randn(10000, 10000)
b_cpu = torch.randn(10000, 10000)

start = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
print(f"CPU Time: {time.time() - start:.4f} seconds")

# Operazione su GPU Intel Arc
a_xpu = a_cpu.to(device)
b_xpu = b_cpu.to(device)

start = time.time()
c_xpu = torch.matmul(a_xpu, b_xpu)
print(f"XPU Time: {time.time() - start:.4f} seconds")