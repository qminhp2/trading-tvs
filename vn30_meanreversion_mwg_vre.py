import backtrader as bt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def calculate_z_score(spread_series, lookback):
    s = pd.Series(spread_series, dtype=float)
    mean = s.rolling(window=lookback).mean()
    std = s.rolling(window=lookback).std()
    return (s - mean) / std

def half_life(spread_series):
    y = pd.Series(spread_series).dropna()
    x = y.shift(1).dropna()
    dy = (y - x).dropna()   
    x = x.loc[dy.index]
    X = sm.add_constant(x.values)
    beta = sm.OLS(dy.values, X).fit().params[1]
    return int(max(1, round(-np.log(2)/beta))) if beta < 0 else 30

def bt_dt(data, idx=0):
    """Backtrader numeric to python datetime"""
    return bt.num2date(data.datetime[idx])

# Cleaning and align data
def load_clean_csv(path):
    df = pd.read_csv(path, parse_dates=[0], dayfirst=True)
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    # Clean close
    df['Close'] = pd.to_numeric(
        df['Close'].astype(str).str.replace(r'[^0-9.\-]', '', regex=True),
        errors='coerce'
    )
    df.dropna(subset=['Close'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df[['Close']]

df_mwg = load_clean_csv('data/mwg.csv')
df_vre = load_clean_csv('data/vre.csv')
df_vn30 = load_clean_csv('data/vn30_ver2.csv')

# align to common date range to avoid holes / phantom bars
start = max(df_mwg.index.min(), df_vre.index.min(), df_vn30.index.min())
end   = min(df_mwg.index.max(), df_vre.index.max(), df_vn30.index.max())
df_mwg = df_mwg.loc[start:end]
df_vre = df_vre.loc[start:end]
df_vn30 = df_vn30.loc[start:end]

for df in [df_mwg, df_vre, df_vn30]:
    df.sort_values('Date', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)   # guard against log/parse artifacts
    df.dropna(subset=['Close'], inplace=True)

common_idx = df_mwg.index.intersection(df_vre.index).intersection(df_vn30.index)
df_mwg = df_mwg.loc[common_idx].copy()
df_vre = df_vre.loc[common_idx].copy()
df_vn30 = df_vn30.loc[common_idx].copy()

# final sanity before feeding Backtrader (cheap & effective)
assert df_mwg['Close'].notna().all() and df_vre['Close'].notna().all() and df_vn30['Close'].notna().all()

# Johansen cointegration test
df_log = pd.concat([
    np.log(df_mwg['Close']).rename('log_MWG'),
    np.log(df_vre['Close']).rename('log_VRE'),
    np.log(df_vn30['Close']).rename('log_VN30')
], axis=1)

df_log = df_log.replace([np.inf, -np.inf], np.nan).dropna()

joh = coint_johansen(df_log.values, det_order=1, k_ar_diff=1)
beta = joh.evec[:, 0]
# Force VN30 coefficient to -1 and keep banks net-positive
if beta[-1] == 0:
    raise ValueError('VN30 coeff is zero in eigenvector')
if (beta[0] + beta[1]) < 0:      # make bank-side positive
    beta = -beta
beta = beta * (-1.0 / beta[-1])  # now beta[-1] == -1.0 exactly

# dynamic params from data

spread_vec = df_log.values @ beta
hl = half_life(spread_vec)
lookback = max(20, min(5 * hl, 120))


z = calculate_z_score(spread_vec, lookback).dropna()
p80, p90, p95 = z.abs().quantile([0.80, 0.90, 0.95]).tolist()
z_long = -p90
z_short = +p90
z_exit = 0.1
z_stop = abs(z_short) + 1

"""
z_long = -2.5
z_short = 2.5
z_exit = 0.1
z_stop =abs(z_short)+1
"""


print("Suggested params from data:",
    dict(z_score_long_entry=z_long,
        z_score_short_entry=z_short,
        z_score_exit=z_exit,
        stop_loss_zscore=z_stop,
        lookback_zscore=lookback))

# Strategy class
class LogSpread(bt.Strategy):
    params = (
        ('z_score_long_entry', z_long),
        ('z_score_short_entry', z_short),
        ('z_score_exit', z_exit),
        ('stop_loss_zscore', z_stop),
        ('lookback_zscore', lookback),
        ('risk_per_trade', 0.01),
        ('printlog', True),
        ('cooldown', 0),  # allow immediate re-entry
    )

    def __init__(self):
        self.mwg, self.vre, self.vn30 = self.datas
        self.mwg_close = self.mwg.close
        self.vre_close = self.vre.close
        self.vn30_close = self.vn30.close

        self.vector = beta
        self.spread_hist = []
        self.zscore = None

        # position state
        self.position_direction = 0   # -1 short-spread, +1 long-spread, 0 flat

        # trade bookkeeping
        self.trade_returns = []
        self.trade_open_value = None
        self.entry_dt = None
        self.exit_dt = None
        self.is_trade_open = False

        # keep refs to open orders to know fills/cancellations
        self.open_orders = []

    # one cooldown clock for the spread (we exit all legs together)
    # Moved to __init__

    # Helper functions
    def log(self, msg):
        if self.p.printlog:
            dt = bt_dt(self.data0)
            print(f'{dt:%Y-%m-%d %H:%M:%S} {msg}')

    def in_position(self):
        # FIX: never rely on self.position with multi-data.
        return any(self.getposition(d).size != 0 for d in (self.mwg, self.vre, self.vn30))

    def flat_now(self):
        return all(self.getposition(d).size == 0 for d in (self.mwg, self.vre, self.vn30))

    def cancel_open_orders(self):
        for o in list(self.open_orders):
            try:
                self.cancel(o)
            except:
                pass
        self.open_orders.clear()

    # Order notifs
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        dname = order.data._name if hasattr(order.data, "_name") else str(order.data)
        if order.status == order.Completed:
            fdt = bt.num2date(order.executed.dt)
            side = 'BUY' if order.isbuy() else 'SELL'
            self.log(f'ORDER FILLED {side} {dname} @ {order.executed.price:.4f} ({fdt:%Y-%m-%d %H:%M:%S})')

            # mark portfolio entry on the FIRST fill from a flat state
            if not self.is_trade_open and not self.position_direction == 0:
                # position_direction was set by signal; this is the first executed leg
                self.trade_open_value = float(self.broker.getvalue())
                self.entry_dt = fdt
                self.is_trade_open = True

            # mark portfolio exit when all legs are flat after any fill
            if self.is_trade_open and self.flat_now():
                exit_val = float(self.broker.getvalue())
                ret = (exit_val / self.trade_open_value) - 1.0 if self.trade_open_value else 0.0
                self.trade_returns.append(ret)
                self.exit_dt = fdt
                self.log(f'TRADE CLOSED @ {self.exit_dt:%Y-%m-%d %H:%M:%S} | Return: {ret:.4%}')
                # reset
                self.trade_open_value = None
                self.entry_dt = None
                self.exit_dt = None
                self.is_trade_open = False
                self.position_direction = 0

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'ORDER {dname} {order.getstatusname()}')
        # cleanup
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            if order in self.open_orders:
                self.open_orders.remove(order)

    def notify_trade(self, trade):
        # optional: log per-leg PnL on close (info only)
        if trade.isclosed:
            dname = trade.data._name if hasattr(trade.data, "_name") else str(trade.data)
            self.log(f'TRADE {dname} closed: pnl={trade.pnlcomm:.2f}')

    # Main logic
    def next(self):
        prices = np.array([
            np.log(self.mwg_close[0]),
            np.log(self.vre_close[0]),
            np.log(self.vn30_close[0])
        ], dtype=float)

        spread = float(np.dot(self.vector, prices))
        self.spread_hist.append(spread)

        if len(self.spread_hist) < self.p.lookback_zscore:
            return

        zscore = float(calculate_z_score(self.spread_hist, self.p.lookback_zscore).iloc[-1])
        self.zscore = zscore
        self.log(f'Spread: {spread:.4f}, Z-score: {zscore:.2f}')

        base = max(1, int(0.1 * self.broker.getvalue() /
                         (self.mwg_close[0] + self.vre_close[0] + abs(self.vn30_close[0]))))

        # Notional hedge for VN30 leg (keeps code simple, no new func)
        vn30_sz = int(round(
            base * abs(self.vector[2]) *
            ((self.mwg_close[0] + self.vre_close[0]) / max(1e-12, self.vn30_close[0]))
        ))
        vn30_sz = max(1, vn30_sz)

        # ENTRY (from flat) - z-score only, no trend/cooldown filter
        if self.flat_now():
            self.position_direction = 0
            self.cancel_open_orders()
            if zscore < self.p.z_score_long_entry:
                self.log('SIGNAL: ENTER LONG SPREAD (+MWG +VRE -VN30)')
                self.position_direction = +1
                self.open_orders += [
                    self.buy(data=self.mwg, size=base),
                    self.buy(data=self.vre, size=base),
                    self.sell(data=self.vn30, size=vn30_sz),
                ]
                return
            if zscore > self.p.z_score_short_entry:
                self.log('SIGNAL: ENTER SHORT SPREAD (-MWG -VRE +VN30)')
                self.position_direction = -1
                self.open_orders += [
                    self.sell(data=self.mwg, size=base),
                    self.sell(data=self.vre, size=base),
                    self.buy(data=self.vn30, size=vn30_sz),
                ]
                return

        # EXIT (while in position)
        if self.position_direction == +1:
            # stop or mean reversion exit
            if (zscore <= -self.p.stop_loss_zscore) or (zscore >= -self.p.z_score_exit):
                self.log('SIGNAL: CLOSE LONG SPREAD')
                self.cancel_open_orders()
                self.open_orders += [
                    self.close(self.mwg),
                    self.close(self.vre),
                    self.close(self.vn30),
                ]
                self._last_exit_bar_idx = len(self)   # cooldown start

        elif self.position_direction == -1:
            if (zscore >= self.p.stop_loss_zscore) or (zscore <= self.p.z_score_exit):
                self.log('SIGNAL: CLOSE SHORT SPREAD')
                self.cancel_open_orders()
                self.open_orders += [
                    self.close(self.mwg),
                    self.close(self.vre),
                    self.close(self.vn30),
                ]
                self._last_exit_bar_idx = len(self)   # cooldown start - marks time of last exit. Ensure strategy doesn't re-enter positions immediately.
        # NOTE: no position_direction reset here, it is done in notify_trade when all legs are flat

# Run strategy
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)  
    cerebro.addstrategy(LogSpread)

    # Named data feeds (names shown in logs)
    data0 = bt.feeds.PandasData(dataname=df_mwg, name='MWG')
    data1 = bt.feeds.PandasData(dataname=df_vre, name='VRE')
    data2 = bt.feeds.PandasData(dataname=df_vn30, name='VN30')


    cerebro.adddata(data0)
    cerebro.adddata(data1)
    cerebro.adddata(data2)

    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # NOTE: If you prefer signal timestamp == fill timestamp, consider:
    # cerebro.broker.set_coc(True)  # cheat-on-open (optional)

    # Daily portfolio returns (broker value)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Days, data=None, _name='timereturns')

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    print('Running backtest...')
    res = cerebro.run()
    strat = res[0]
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # ----- Sharpe (daily) -----
    tr = strat.analyzers.timereturns.get_analysis()  # dict {dt: ret}
    daily_ret = np.array(list(tr.values()), dtype=float)
    daily_ret = daily_ret[np.isfinite(daily_ret)]
    if daily_ret.size > 1 and np.std(daily_ret, ddof=1) > 0:
        daily_sharpe = np.mean(daily_ret) / np.std(daily_ret, ddof=1)
        ann_sharpe = daily_sharpe * np.sqrt(252.0)
        print(f'Daily Sharpe: {daily_sharpe:.4f} | Annualized: {ann_sharpe:.4f}')
    else:
        print('Daily Sharpe: N/A (insufficient or zero-variance returns)')

    # ----- Sharpe (per-trade) -----
    per_trade = np.array(getattr(strat, 'trade_returns', []), dtype=float)
    per_trade = per_trade[np.isfinite(per_trade)]
    if per_trade.size > 1 and np.std(per_trade, ddof=1) > 0:
        per_trade_sharpe = np.mean(per_trade) / np.std(per_trade, ddof=1)
        print(f'Per-Trade Sharpe: {per_trade_sharpe:.4f} | Num trades: {per_trade.size}, Avg trade: {np.mean(per_trade):.4%}')
    else:
        print('Per-Trade Sharpe: N/A (insufficient or zero-variance trade returns)')

    print(start)
    print(end)

    cerebro.plot(style='candlestick', barup='green', bardown='red', volume=False)