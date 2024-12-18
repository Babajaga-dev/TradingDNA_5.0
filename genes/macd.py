import talib

class MACDGene:
    def __init__(self, config):
        self.fastperiod = config['indicators']['macd']['fastperiod']
        self.slowperiod = config['indicators']['macd']['slowperiod']
        self.signalperiod = config['indicators']['macd']['signalperiod']

    def apply(self, data):
        macd, signal, _ = talib.MACD(data['close'], 
                                     fastperiod=self.fastperiod, 
                                     slowperiod=self.slowperiod, 
                                     signalperiod=self.signalperiod)
        return macd, signal
