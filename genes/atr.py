import pandas as pd

class ATRGene:
    def __init__(self, config):
        atr_config = config['indicators']['atr']
        self.period = atr_config['period']

    def apply(self, data):
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        return atr
