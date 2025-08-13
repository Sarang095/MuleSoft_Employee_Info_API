from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
	"""Average True Range using simple moving average of True Range.
	Requires columns: high, low, close.
	"""
	high = df["high"].astype(float)
	low = df["low"].astype(float)
	close = df["close"].astype(float)
	prev_close = close.shift(1)
	tr1 = (high - low).abs()
	tr2 = (high - prev_close).abs()
	tr3 = (low - prev_close).abs()
	tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
	atr = tr.rolling(window=period, min_periods=period).mean()
	return atr


def project_sl_tp(
	close_price: float,
	atr_value: float,
	sl_mult: float = 2.0,
	tp_mult: float = 4.0,
	direction: Literal["long", "short"] = "long",
) -> Tuple[float, float]:
	"""Return (stop_loss_price, take_profit_price)."""
	if not np.isfinite(atr_value):
		return (np.nan, np.nan)
	if direction == "long":
		return (close_price - sl_mult * atr_value, close_price + tp_mult * atr_value)
	else:
		return (close_price + sl_mult * atr_value, close_price - tp_mult * atr_value)


def position_size_by_risk(
	equity: float,
	entry_price: float,
	stop_price: float,
	risk_pct: float,
) -> float:
	"""Units sized so that loss at stop equals equity * risk_pct.
	Returns 0 if parameters invalid.
	"""
	price_risk = abs(entry_price - stop_price)
	if price_risk <= 0 or not np.isfinite(price_risk):
		return 0.0
	amount_at_risk = equity * float(risk_pct)
	units = amount_at_risk / price_risk
	return max(float(units), 0.0)