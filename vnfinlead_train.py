
import backtrader as bt
from backtrader.indicators import BollingerBands
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class CustomSpread(bt.Indicator):
    lines = ('spread',)
    def __init__(self):
        self.lines.spread = (self.data0.close + self.data1.close) / 2.0 - self.data2.close # Calculate spread as average of two banks minus FUESSVFL price

class SecondStrat(bt.Strategy):
    params = dict(
        stop_loss=0.05,  # 5% stop loss
        take_profit=0.10  # 10% take profit
    )

    def __init__(self):
        self.tcb = self.datas[0]
        self.stb = self.datas[1]
        self.fue = self.datas[2]
        self.spread = CustomSpread(self.tcb, self.stb, self.fue)
        self.boll = BollingerBands(self.spread, period=50, devfactor=2.8)

        self.entry_spread = None # Track entry spread for stop-loss/take-profit calculations
        self.position_type = None  # 'long', 'short', or None
        self.trade_log = [] # Store records of trades

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            action = 'BUY' if order.isbuy() else 'SELL'
            self.log(f'{action} {order.data._name} EXECUTED at {order.executed.price:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order {order.data._name} Canceled/Margin/Rejected')

    def next(self):
        spread_val = self.spread[0]
        if not self.position_type:
            # ENTRY CONDITIONS:
            if spread_val < self.boll.lines.bot[0]: # spread < lower band. Long banks, short FUESSVFL
                self.buy(data=self.tcb, size=1)
                self.buy(data=self.stb, size=1)
                self.sell(data=self.fue, size=2)
                self.entry_spread = spread_val
                self.position_type = 'long'
                self.log(f'ENTRY: LONG banks, SHORT FUESSVFL at spread={spread_val:.2f}')
            elif spread_val > self.boll.lines.top[0]: #spread > upper band. Short banks, long FUESSVFL
                self.sell(data=self.tcb, size=1)
                self.sell(data=self.stb, size=1)
                self.buy(data=self.fue, size=2)
                self.entry_spread = spread_val
                self.position_type = 'short'
                self.log(f'ENTRY: SHORT banks, LONG FUESSVFL at spread={spread_val:.2f}')

        else:
            # EXIT CONDITIONS: Convergence (spread reverts within bands) post-breach
            if self.boll.lines.bot[0] < spread_val < self.boll.lines.top[0]:
                self.close(data=self.tcb, size=1)
                self.close(data=self.stb, size=1)
                if self.position_type == 'long':
                    self.sell(data=self.fue, size=2) # Sell FUESSVFL to close the short
                elif self.position_type == 'short':
                    self.buy(data=self.fue, size=2) # Buy FUESSVFL to close the long
                self.log(f'EXIT: Spread reverted to band center at spread={spread_val:.2f}, exited all positions')
                self.position_type = None
                self.entry_spread = None
                return

            # Stop-loss and take-profit logic
            if self.position_type == 'long': # Calculate the change from entry spread
                change = (spread_val - self.entry_spread) / abs(self.entry_spread) # Long position --> want spread to increase
            else:
                change = (self.entry_spread - spread_val) / abs(self.entry_spread) # Short position --> want spread to decrease

            if change <= -self.params.stop_loss: # Stop loss: spread drops below entry by 5% --> close
                self.close(data=self.tcb)
                self.close(data=self.stb)
                self.close(data=self.fue)
                self.log(f'EXIT: Stop-loss hit with spread change {change:.2%}')
                self.position_type = None
                self.entry_spread = None

            elif change >= self.params.take_profit: #Take profit: spread increases above entry by 10% --> close
                self.close(data=self.tcb)
                self.close(data=self.stb)
                self.close(data=self.fue)
                self.log(f'EXIT: Take-profit hit with spread change {change:.2%}')
                self.position_type = None
                self.entry_spread = None
            
    def notify_trade(self, trade):
        if trade.isclosed:
            dt = self.data.datetime.date(0)
            entry_price = trade.price

            self.log(
                f'TRADE CLOSED: {trade.data._name}, '
                f'PNL GROSS={trade.pnl:.2f}, PNL NET={trade.pnlcomm:.2f}, BARLEN={trade.barlen}'
            )

            self.trade_log.append({
                'Date': dt,
                'Instrument': trade.data._name,
                'Entry Price': f'{entry_price:.2f}',
                'PnL Gross': trade.pnl,
                'PnL Net': trade.pnlcomm,
                'Bars Held': trade.barlen
            })


    def stop(self):
        df = pd.DataFrame(self.trade_log)
        df.to_csv('trade_log.csv', index=False)
        print(f"\nTrade log exported to trade_log.csv with {len(df)} trades.")



if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SecondStrat)

    # Define split ranges
    
    date_start = pd.to_datetime('2020-03-19')
    date_end = pd.to_datetime('2025-07-21')


    # Load full datasets as Pandas DataFrames
    df_fue = pd.read_csv('data/fuess_final.csv', parse_dates=[0], dayfirst=True)
    df_fue = df_fue.sort_values(df_fue.columns[0])  # Sort by date ascending
    df_tcb = pd.read_csv('data/tcb_final.csv', parse_dates=[0], dayfirst=True)
    df_tcb = df_tcb.sort_values(df_tcb.columns[0])  # Sort by date ascending
    df_stb = pd.read_csv('data/stb_final.csv', parse_dates=[0], dayfirst=True)
    df_stb = df_stb.sort_values(df_stb.columns[0])  # Sort by date ascending

    # Filter to train only
    df_fue_train = df_fue[(df_fue.iloc[:, 0] >= date_start) & (df_fue.iloc[:, 0] <= date_end)]
    df_tcb_train = df_tcb[(df_tcb.iloc[:, 0] >= date_start) & (df_tcb.iloc[:, 0] <= date_end)]
    df_stb_train = df_stb[(df_stb.iloc[:, 0] >= date_start) & (df_stb.iloc[:, 0] <= date_end)]

    # Define PandasData class
    class PandasData(bt.feeds.PandasData):
        params = (
            ('datetime', 0),
            ('open', 2),
            ('high', 3),
            ('low', 4),
            ('close', 1),
            ('volume', 5),
            ('openinterest', -1),
        )

    # Add data to Cerebro
    cerebro.adddata(PandasData(dataname=df_tcb_train), name='TCB')
    cerebro.adddata(PandasData(dataname=df_stb_train), name='STB')
    cerebro.adddata(PandasData(dataname=df_fue_train), name='FUESSVFL')

    # Broker setup
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.Trades)

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.run(cheat_on_open=True)
    print(f'Ending Portfolio Value: {cerebro.broker.getvalue():.2f}')

    cerebro.plot(style='candlestick', barup='green', bardown='red', volume=False)