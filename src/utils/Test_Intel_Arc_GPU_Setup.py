import torch

# Rilevazione dispositivo
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")

# Test operazioni sulla GPU (o CPU se XPU non è disponibile)
try:
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = x + y
    print(f"Result on {device}: {z}")
except Exception as e:
    print(f"Error using device {device}: {e}")
