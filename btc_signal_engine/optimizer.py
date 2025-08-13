from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

from .engine import backtest_strategy


@dataclass
class OptimizeResult:
	best_params: Dict[str, float]
	best_metrics: Dict[str, float]


def _default_space(strategy: str) -> Dict[str, Tuple[float, float, int]]:
	if strategy == "sma":
		return {"fast": (5, 50, 1), "slow": (20, 200, 1)}
	elif strategy == "rsi":
		return {"rsi_period": (5, 30, 1), "rsi_low": (10, 40, 1), "rsi_high": (60, 90, 1)}
	elif strategy == "macd":
		return {"fast": (6, 20, 1), "slow": (15, 40, 1), "macd_signal": (5, 15, 1)}
	elif strategy == "bollinger":
		return {"bb_window": (10, 40, 1), "bb_mult": (1, 3, 1)}
	else:
		return {}


def _sample(space: Dict[str, Tuple[float, float, int]]) -> Dict[str, float]:
	params: Dict[str, float] = {}
	for name, (lo, hi, step) in space.items():
		if isinstance(step, int) and step >= 1:
			val = random.randrange(int(lo), int(hi) + 1, int(step))
		else:
			val = random.uniform(float(lo), float(hi))
		params[name] = float(val)
	return params


def optimize_strategy(
	df: pd.DataFrame,
	strategy: Literal["sma", "rsi", "macd", "bollinger"],
	space: Dict[str, Tuple[float, float, int]] | None = None,
	n_trials: int = 50,
	objective: Literal["sharpe_like", "total_return", "profit_factor"] = "sharpe_like",
	seed: int | None = 42,
	# risk/cost params
	account_size: float = 10000.0,
	trade_pct: float = 1.0,
	risk_pct: float = 0.01,
	leverage: float = 1.0,
	commission_bps: float = 0.0,
	spread_bps: float = 0.0,
	size_mode: Literal["fixed_fraction", "atr_risk"] = "fixed_fraction",
) -> OptimizeResult:
	rnd = random.Random(seed)
	space = space or _default_space(strategy)

	best_score = -float("inf")
	best_params: Dict[str, float] = {}
	best_metrics: Dict[str, float] = {}

	for _ in range(int(n_trials)):
		trial_params = _sample(space)
		metrics = backtest_strategy(
			df=df,
			strategy=strategy,
			fast=int(trial_params.get("fast", 20)),
			slow=int(trial_params.get("slow", 50)),
			macd_signal=int(trial_params.get("macd_signal", 9)),
			rsi_period=int(trial_params.get("rsi_period", 14)),
			rsi_low=float(trial_params.get("rsi_low", 30.0)),
			rsi_high=float(trial_params.get("rsi_high", 70.0)),
			bb_window=int(trial_params.get("bb_window", 20)),
			bb_mult=float(trial_params.get("bb_mult", 2.0)),
			account_size=account_size,
			trade_pct=trade_pct,
			risk_pct=risk_pct,
			leverage=leverage,
			commission_bps=commission_bps,
			spread_bps=spread_bps,
			size_mode=size_mode,
		)
		if objective == "total_return":
			score = metrics.get("total_return", 0.0)
		elif objective == "profit_factor":
			score = metrics.get("profit_factor", 0.0)
		else:
			score = metrics.get("sharpe_like", 0.0)

		if score > best_score:
			best_score = score
			best_params = trial_params
			best_metrics = metrics

	return OptimizeResult(best_params=best_params, best_metrics=best_metrics)