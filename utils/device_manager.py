import torch

class DeviceManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return self.device
