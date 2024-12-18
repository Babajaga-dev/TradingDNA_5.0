import pandas as pd

class StochasticOscillatorGene:
    def __init__(self, config):
        stoch_config = config['indicators']['stochastic']
        self.k_period = stoch_config['k_period']
        self.d_period = stoch_config['d_period']
        self.smooth = stoch_config['smooth']

    def apply(self, data):
        # Calcolo %K
        high_max = data['high'].rolling(window=self.k_period).max()
        low_min = data['low'].rolling(window=self.k_period).min()
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        
        # Applico smoothing a %K
        if self.smooth > 1:
            k = k.rolling(window=self.smooth).mean()
        
        # Calcolo %D (media mobile di %K)
        d = k.rolling(window=self.d_period).mean()
        
        return k, d
