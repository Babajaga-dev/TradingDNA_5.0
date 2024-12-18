import pandas as pd
import numpy as np

class SimulationProcessor:
    def __init__(self, config):
        self.position_size_pct = config['trading']['position']['size_pct'] / 100
        self.stop_loss_pct = config['trading']['position']['stop_loss_pct'] / 100
        self.take_profit_pct = config['trading']['position']['take_profit_pct'] / 100
        self.initial_capital = config['trading']['initial_capital']

    def run_simulation(self, data, signals):
        equity = self.initial_capital
        positions = []
        pnl = []

        for i in range(len(signals)):
            price = data['close'].iloc[i]
            signal = signals.iloc[i]

            if signal > 0 and not positions:
                position_size = equity * self.position_size_pct
                positions.append({'entry_price': price, 'size': position_size})
                print(f"Opened position at {price}")

            elif signal < 0 and positions:
                for pos in positions:
                    pnl_value = (price - pos['entry_price']) * pos['size'] / pos['entry_price']
                    pnl.append(pnl_value)
                    equity += pnl_value
                positions.clear()
                print(f"Closed all positions at {price}")

        return pnl, equity
