from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import backtrader as bt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import config
from feature_engineer import FeatureEngineer
from strategy_engine import StrategyEngine
from risk_manager import RiskManager
from data_handler import DataHandler
from logger import get_logger


class FeaturePandasData(bt.feeds.PandasData):
    lines = ("ema21", "ema50", "ema200", "adx14", "rsi14", "bbl", "bbm", "bbu", "macd", "macds", "macdh", "atr14")
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
        ("ema21", "EMA21"),
        ("ema50", "EMA50"),
        ("ema200", "EMA200"),
        ("adx14", "ADX14"),
        ("rsi14", "RSI14"),
        ("bbl", "BBL"),
        ("bbm", "BBM"),
        ("bbu", "BBU"),
        ("macd", "MACD"),
        ("macds", "MACD_SIGNAL"),
        ("macdh", "MACD_HIST"),
        ("atr14", "ATR14"),
    )


class RegimeStrategyBT(bt.Strategy):
    params = dict(
        symbol=config.SYMBOL,
    )

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.engine = StrategyEngine()
        self.risk = RiskManager()
        self.order = None

        # State
        self.in_position = False
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0

        # Tracking
        self.starting_value = None
        self.value_series: List[Tuple[pd.Timestamp, float]] = []
        self.buy_signals: List[pd.Timestamp] = []
        self.sell_signals: List[pd.Timestamp] = []

    def start(self):
        self.starting_value = self.broker.getvalue()

    def log(self, txt):
        self.logger.info(txt)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}"
                )
            else:  # Sell
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}"
                )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        self.order = None

    def next(self):
        dt = bt.num2date(self.data.datetime[0])
        self.value_series.append((pd.Timestamp(dt), float(self.broker.getvalue())))

        # Build a tiny DataFrame for engine
        cols = [
            "open","high","low","close","volume","EMA21","EMA50","EMA200","ADX14","RSI14","BBL","BBM","BBU","MACD","MACD_SIGNAL","MACD_HIST","ATR14"
        ]
        cur = {
            "open": float(self.data.open[0]),
            "high": float(self.data.high[0]),
            "low": float(self.data.low[0]),
            "close": float(self.data.close[0]),
            "volume": float(self.data.volume[0]),
            "EMA21": float(self.data.ema21[0]),
            "EMA50": float(self.data.ema50[0]),
            "EMA200": float(self.data.ema200[0]),
            "ADX14": float(self.data.adx14[0]),
            "RSI14": float(self.data.rsi14[0]),
            "BBL": float(self.data.bbl[0]),
            "BBM": float(self.data.bbm[0]),
            "BBU": float(self.data.bbu[0]),
            "MACD": float(self.data.macd[0]),
            "MACD_SIGNAL": float(self.data.macds[0]),
            "MACD_HIST": float(self.data.macdh[0]),
            "ATR14": float(self.data.atr14[0]),
        }
        prev = {
            "open": float(self.data.open[-1]),
            "high": float(self.data.high[-1]),
            "low": float(self.data.low[-1]),
            "close": float(self.data.close[-1]),
            "volume": float(self.data.volume[-1]),
            "EMA21": float(self.data.ema21[-1]),
            "EMA50": float(self.data.ema50[-1]),
            "EMA200": float(self.data.ema200[-1]),
            "ADX14": float(self.data.adx14[-1]),
            "RSI14": float(self.data.rsi14[-1]),
            "BBL": float(self.data.bbl[-1]),
            "BBM": float(self.data.bbm[-1]),
            "BBU": float(self.data.bbu[-1]),
            "MACD": float(self.data.macd[-1]),
            "MACD_SIGNAL": float(self.data.macds[-1]),
            "MACD_HIST": float(self.data.macdh[-1]),
            "ATR14": float(self.data.atr14[-1]),
        }
        df = pd.DataFrame([prev, cur], columns=cols)

        decision = self.engine.generate_signal(df)

        if not self.position:
            if decision.signal == "BUY":
                atr = cur["ATR14"]
                cash = self.broker.getvalue()
                entry_price = float(self.data.close[0])
                order_spec = self.risk.build_order(config.SYMBOL, entry_price, atr, cash)
                size = order_spec.quantity
                if size <= 0:
                    return
                # Place market buy
                self.order = self.buy(size=size)
                self.in_position = True
                self.position_size = size
                self.entry_price = entry_price
                self.stop_loss = order_spec.stop_loss
                self.take_profit = order_spec.take_profit
                self.buy_signals.append(pd.Timestamp(dt))
        else:
            # Exit on SELL signal
            if decision.signal == "SELL":
                self.order = self.sell(size=self.position.size)
                self.in_position = False
                self.sell_signals.append(pd.Timestamp(dt))
            else:
                # Manage ATR trailing & TP
                price = float(self.data.close[0])
                atr = cur["ATR14"]
                new_sl = price - config.ATR_MULTIPLIER * atr
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl
                if price <= self.stop_loss or price >= self.take_profit:
                    self.order = self.sell(size=self.position.size)
                    self.in_position = False
                    self.sell_signals.append(pd.Timestamp(dt))

    def stop(self):
        # Ensure position closed at end
        if self.position:
            self.close()


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    equity: pd.Series
    price: pd.Series
    buys: List[pd.Timestamp]
    sells: List[pd.Timestamp]


def run_backtest(start: str, end: str, plot_name: Optional[str] = None) -> BacktestResult:
    logger = get_logger("Backtest")

    os.makedirs(config.PLOT_OUTPUT, exist_ok=True)

    # Fetch and prepare data
    dh = DataHandler()
    raw = dh.fetch_ohlcv(config.SYMBOL, timeframe=config.TIMEFRAME, since=start, until=end)

    fe = FeatureEngineer()
    df = fe.add_features(raw)

    # Prepare Backtrader
    cerebro = bt.Cerebro()
    data = FeaturePandasData(dataname=df)
    cerebro.adddata(data)

    cerebro.addstrategy(RegimeStrategyBT)

    # Broker settings
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=config.TRADING_FEE)
    cerebro.broker.set_slippage_perc(perc=config.SLIPPAGE_PCT)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', tann=252)

    logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    strategies = cerebro.run()
    strat = strategies[0]
    final_value = cerebro.broker.getvalue()
    logger.info(f"Final Portfolio Value: {final_value:.2f}")

    # Extract analyzers
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', float('nan'))
    dd = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    rets = strat.analyzers.returns.get_analysis()

    total_trades = (trades.get('total', {}) or {}).get('total', 0)
    won = (trades.get('won', {}) or {}).get('total', 0)
    lost = (trades.get('lost', {}) or {}).get('total', 0)
    win_rate = (won / total_trades * 100.0) if total_trades else 0.0

    gross_profit = float((trades.get('pnl', {}) or {}).get('gross', {}).get('profit', 0.0))
    gross_loss = float((trades.get('pnl', {}) or {}).get('gross', {}).get('loss', 0.0))
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float('inf')

    starting_value = 10000.0
    net_profit = final_value - starting_value
    ann_return = float(rets.get('rnorm100', float('nan')))  # annualized return in percent

    max_dd = float(dd.get('max', {}).get('drawdown', float('nan')))  # percent

    metrics = {
        "Starting Value": starting_value,
        "Final Value": final_value,
        "Net Profit": net_profit,
        "Annualized Return %": ann_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown %": max_dd,
        "Total Trades": total_trades,
        "Win Rate %": win_rate,
        "Profit Factor": profit_factor,
    }

    # Build equity series
    values = pd.Series({ts: val for ts, val in strat.value_series}).sort_index()
    price = df['close']

    # Plot
    if plot_name:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(price.index, price.values, color='steelblue', label='BTC/USDT Price')
        ax1.set_ylabel('Price')
        ax2 = ax1.twinx()
        ax2.plot(values.index, values.values, color='darkorange', label='Portfolio Value')
        ax2.set_ylabel('Portfolio Value')

        # Markers
        for ts in strat.buy_signals:
            ax1.axvline(ts, color='green', alpha=0.2)
        for ts in strat.sell_signals:
            ax1.axvline(ts, color='red', alpha=0.2)

        ax1.set_title(f"Backtest {start} to {end}")
        fig.tight_layout()
        outpath = os.path.join(config.PLOT_OUTPUT, plot_name)
        fig.savefig(outpath)

    return BacktestResult(
        metrics=metrics,
        equity=values,
        price=price,
        buys=strat.buy_signals,
        sells=strat.sell_signals,
    )


def run_in_sample_out_of_sample() -> Tuple[BacktestResult, BacktestResult]:
    train = run_backtest(config.TRAIN_START, config.TRAIN_END, plot_name="equity_train.png")
    test = run_backtest(config.TEST_START, config.TEST_END, plot_name="equity_test.png")
    return train, test