import pandas as pd

class VolatilityGene:
    def __init__(self, config):
        volatility_config = config['indicators']['volatility']
        self.window = volatility_config['window']

    def apply(self, data):
        return data['close'].pct_change().rolling(window=self.window).std()
