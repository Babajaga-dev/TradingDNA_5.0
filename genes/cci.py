import pandas as pd

class CCIGene:
    def __init__(self, config):
        cci_config = config['indicators']['cci']
        self.period = cci_config['period']

    def apply(self, data):
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        ma = typical_price.rolling(window=self.period).mean()
        mad = (typical_price - ma).abs().rolling(window=self.period).mean()
        cci = (typical_price - ma) / (0.015 * mad)
        return cci
