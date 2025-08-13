from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

Signal = Literal["buy", "sell", "hold"]


@dataclass
class StrategyParams:
	fast: int = 20
	slow: int = 50
	macd_signal: int = 9
	rsi_period: int = 14
	rsi_low: float = 30.0
	rsi_high: float = 70.0
	bb_window: int = 20
	bb_mult: float = 2.0


def _ema(series: pd.Series, span: int) -> pd.Series:
	return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
	return series.rolling(window=window, min_periods=window).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
	# Wilder's RSI
	delta = series.diff()
	gain = np.where(delta > 0, delta, 0.0)
	loss = np.where(delta < 0, -delta, 0.0)
	gain_series = pd.Series(gain, index=series.index)
	loss_series = pd.Series(loss, index=series.index)
	avg_gain = gain_series.ewm(alpha=1 / period, adjust=False).mean()
	avg_loss = loss_series.ewm(alpha=1 / period, adjust=False).mean()
	rs = avg_gain / (avg_loss.replace(0, np.nan))
	rsi = 100 - (100 / (1 + rs))
	return rsi.fillna(50.0)


def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
	ema_fast = _ema(series, fast)
	ema_slow = _ema(series, slow)
	macd_line = ema_fast - ema_slow
	signal_line = _ema(macd_line, signal)
	hist = macd_line - signal_line
	return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def _bollinger(series: pd.Series, window: int, mult: float) -> pd.DataFrame:
	ma = _sma(series, window)
	std = series.rolling(window=window, min_periods=window).std()
	upper = ma + mult * std
	lower = ma - mult * std
	return pd.DataFrame({"bb_mid": ma, "bb_upper": upper, "bb_lower": lower})


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
	if not df.index.is_monotonic_increasing:
		df = df.sort_index()
	return df


def _validate_ohlcv(df: pd.DataFrame) -> None:
	required = ["open", "high", "low", "close", "volume"]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Input DataFrame missing required columns: {missing}")
	if not isinstance(df.index, pd.DatetimeIndex):
		raise ValueError("DataFrame index must be a DatetimeIndex")


def compute_indicators(
	df: pd.DataFrame,
	params: StrategyParams,
) -> pd.DataFrame:
	"""Compute all indicators used by supported strategies and attach as columns."""
	_validate_ohlcv(df)
	df = _ensure_sorted(df).copy()
	close = df["close"]

	# SMA fast/slow
	df["sma_fast"] = _sma(close, params.fast)
	df["sma_slow"] = _sma(close, params.slow)

	# RSI
	df["rsi"] = _rsi(close, params.rsi_period)

	# MACD
	macd_df = _macd(close, params.fast, params.slow, params.macd_signal)
	df = df.join(macd_df)

	# Bollinger Bands
	bb_df = _bollinger(close, params.bb_window, params.bb_mult)
	df = df.join(bb_df)

	return df


def _cross_over(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
	prev = series_a.shift(1) <= series_b.shift(1)
	now = series_a > series_b
	return prev & now


def _cross_under(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
	prev = series_a.shift(1) >= series_b.shift(1)
	now = series_a < series_b
	return prev & now


def generate_signals(
	df: pd.DataFrame,
	strategy: Literal["sma", "rsi", "macd", "bollinger"],
	params: StrategyParams,
) -> pd.DataFrame:
	"""Generate strategy signals per bar: buy/sell/hold in a new 'signal' column.
	Signals are generated on close of each bar.
	"""
	df = compute_indicators(df, params).copy()

	buy = pd.Series(False, index=df.index)
	sell = pd.Series(False, index=df.index)

	if strategy == "sma":
		buy = _cross_over(df["sma_fast"], df["sma_slow"])
		sell = _cross_under(df["sma_fast"], df["sma_slow"])
	elif strategy == "rsi":
		buy = _cross_over(df["rsi"], pd.Series(params.rsi_low, index=df.index))
		sell = _cross_under(df["rsi"], pd.Series(params.rsi_high, index=df.index))
	elif strategy == "macd":
		buy = _cross_over(df["macd"], df["macd_signal"]) 
		sell = _cross_under(df["macd"], df["macd_signal"]) 
	elif strategy == "bollinger":
		# Mean reversion: re-enter when price crosses back inside bands
		buy = _cross_over(df["close"], df["bb_lower"]) 
		sell = _cross_under(df["close"], df["bb_upper"]) 
	else:
		raise ValueError(f"Unknown strategy: {strategy}")

	signal = pd.Series("hold", index=df.index, dtype="object")
	signal[buy] = "buy"
	signal[sell] = "sell"
	df["signal"] = signal
	return df


def evaluate_latest_signal(
	df: pd.DataFrame,
	strategy: Literal["sma", "rsi", "macd", "bollinger"],
	fast: int = 20,
	slow: int = 50,
	macd_signal: int = 9,
	rsi_period: int = 14,
	rsi_low: float = 30.0,
	rsi_high: float = 70.0,
	bb_window: int = 20,
	bb_mult: float = 2.0,
) -> Dict[str, str]:
	params = StrategyParams(
		fast=fast,
		slow=slow,
		macd_signal=macd_signal,
		rsi_period=rsi_period,
		rsi_low=rsi_low,
		rsi_high=rsi_high,
		bb_window=bb_window,
		bb_mult=bb_mult,
	)
	df_signals = generate_signals(df, strategy, params)
	last_idx = df_signals.index[-1]
	last_signal = str(df_signals.loc[last_idx, "signal"]) 

	reason = ""
	if strategy == "sma":
		f_now, s_now = df_signals.loc[last_idx, ["sma_fast", "sma_slow"]]
		prev_idx = df_signals.index[-2] if len(df_signals) >= 2 else last_idx
		f_prev, s_prev = df_signals.loc[prev_idx, ["sma_fast", "sma_slow"]]
		if np.isfinite(f_now) and np.isfinite(s_now):
			if f_prev <= s_prev and f_now > s_now:
				reason = f"fast SMA crossed above slow SMA ({f_now:.2f}>{s_now:.2f})"
			elif f_prev >= s_prev and f_now < s_now:
				reason = f"fast SMA crossed below slow SMA ({f_now:.2f}<{s_now:.2f})"
	elif strategy == "rsi":
		rsi_now = float(df_signals.loc[last_idx, "rsi"]) 
		if rsi_now < rsi_low:
			reason = f"RSI {rsi_now:.1f} < {rsi_low}"
		elif rsi_now > rsi_high:
			reason = f"RSI {rsi_now:.1f} > {rsi_high}"
	elif strategy == "macd":
		m_now, s_now = df_signals.loc[last_idx, ["macd", "macd_signal"]]
		if np.isfinite(m_now) and np.isfinite(s_now):
			reason = f"MACD {m_now:.4f} vs Signal {s_now:.4f}"
	elif strategy == "bollinger":
		c, lo, hi = df_signals.loc[last_idx, ["close", "bb_lower", "bb_upper"]]
		if np.isfinite(lo) and c <= lo:
			reason = f"Close near/below lower band ({c:.2f} <= {lo:.2f})"
		elif np.isfinite(hi) and c >= hi:
			reason = f"Close near/above upper band ({c:.2f} >= {hi:.2f})"

	if not reason:
		reason = "no decisive condition; holding"

	return {"signal": last_signal, "reason": reason}


def backtest_strategy(
	df: pd.DataFrame,
	strategy: Literal["sma", "rsi", "macd", "bollinger"],
	fast: int = 20,
	slow: int = 50,
	macd_signal: int = 9,
	rsi_period: int = 14,
	rsi_low: float = 30.0,
	rsi_high: float = 70.0,
	bb_window: int = 20,
	bb_mult: float = 2.0,
) -> Dict[str, float]:
	params = StrategyParams(
		fast=fast,
		slow=slow,
		macd_signal=macd_signal,
		rsi_period=rsi_period,
		rsi_low=rsi_low,
		rsi_high=rsi_high,
		bb_window=bb_window,
		bb_mult=bb_mult,
	)
	df_sig = generate_signals(df, strategy, params)
	df_sig = df_sig.copy()

	# Simulate: trade on next bar open after a signal
	opens = df_sig["open"]
	closes = df_sig["close"]
	signals = df_sig["signal"].values
	index = df_sig.index

	position = 0  # 0 flat, 1 long
	entry_price = 0.0
	pnl_list = []
	trade_returns = []
	trade_outcomes = []  # 1 win, 0 loss
	equity = 1.0
	equity_curve = []

	for i in range(len(df_sig) - 1):
		bar_signal = signals[i]
		next_open = float(opens.iloc[i + 1])
		close_price = float(closes.iloc[i])

		# Record equity mark-to-market
		if position == 1 and entry_price > 0:
			unreal = (close_price - entry_price) / entry_price
			equity_curve.append(equity * (1 + unreal))
		else:
			equity_curve.append(equity)

		if bar_signal == "buy" and position == 0:
			position = 1
			entry_price = next_open
		elif bar_signal == "sell" and position == 1:
			ret = (next_open - entry_price) / entry_price
			trade_returns.append(ret)
			trade_outcomes.append(1 if ret > 0 else 0)
			equity *= 1 + ret
			pnl_list.append(ret)
			position = 0
			entry_price = 0.0

	# Close any open position at last close
	if position == 1 and entry_price > 0:
		final_ret = (float(closes.iloc[-1]) - entry_price) / entry_price
		trade_returns.append(final_ret)
		trade_outcomes.append(1 if final_ret > 0 else 0)
		equity *= 1 + final_ret
		pnl_list.append(final_ret)
		position = 0
		entry_price = 0.0
		equity_curve.append(equity)
	else:
		if len(equity_curve) < len(df_sig):
			equity_curve.append(equity)

	equity_series = pd.Series(equity_curve, index=index[: len(equity_curve)])
	returns_series = equity_series.pct_change().fillna(0.0)

	total_return = equity - 1.0
	num_trades = len(trade_returns)
	win_rate = float(np.mean(trade_outcomes)) if trade_outcomes else 0.0
	avg_trade_return = float(np.mean(trade_returns)) if trade_returns else 0.0

	# Max drawdown
	roll_max = equity_series.cummax()
	drawdown = (equity_series - roll_max) / roll_max.replace(0, np.nan)
	max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

	# Simple annualization guess: use bars per year if we can infer interval from index
	# Fallback to sharpe over bar returns times sqrt(252)
	ret_std = float(returns_series.std())
	if ret_std > 0:
		sharpe = float((returns_series.mean() / ret_std) * np.sqrt(252))
	else:
		sharpe = 0.0

	return {
		"total_return": float(total_return),
		"num_trades": int(num_trades),
		"win_rate": float(win_rate),
		"avg_trade_return": float(avg_trade_return),
		"max_drawdown": float(max_drawdown),
		"sharpe_like": float(sharpe),
	}