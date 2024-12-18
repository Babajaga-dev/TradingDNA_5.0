import talib

class RSIGene:
    def __init__(self, config):
        self.period = config['indicators']['rsi']['period']

    def apply(self, data):
        return talib.RSI(data['close'], timeperiod=self.period)
