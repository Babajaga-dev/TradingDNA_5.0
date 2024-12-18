import talib

class MovingAverageGene:
    def __init__(self, period=14):
        self.period = period

    def apply(self, data):
        return talib.SMA(data['close'], timeperiod=self.period)
