from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .data import fetch_ohlcv
from .engine import StrategyParams, compute_indicators, evaluate_latest_signal
from .optimizer import optimize_strategy
from .risk import compute_atr, position_size_by_risk, project_sl_tp


@dataclass
class PaperPosition:
	symbol: str
	units: float
	entry_price: float
	entry_time: str
	sl: float
	tp: float
	notional: float
	strategy: str
	params: Dict[str, float]


@dataclass
class PaperConfig:
	symbols: List[str]
	strategy: Literal["sma", "rsi", "macd", "bollinger"]
	interval: str
	limit: int
	account_size: float
	trade_pct: float
	risk_pct: float
	leverage: float
	commission_bps: float
	spread_bps: float
	size_mode: Literal["fixed_fraction", "atr_risk"]
	atr_period: int
	sl_atr_mult: float
	tp_atr_mult: float
	optimize_every: int
	objective: Literal["sharpe_like", "total_return", "profit_factor"]
	log_dir: str
	refresh_seconds: Optional[int] = None
	max_iterations: Optional[int] = None


def _interval_to_seconds(interval: str) -> int:
	mapping = {"1m":60, "5m":300, "15m":900, "1h":3600, "4h":14400, "1d":86400}
	return mapping.get(interval, 60)


def _utcnow_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _ensure_logs(log_dir: str) -> Dict[str, str]:
	os.makedirs(log_dir, exist_ok=True)
	trades_path = os.path.join(log_dir, "trades.csv")
	if not os.path.exists(trades_path):
		with open(trades_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow([
				"timestamp","symbol","action","strategy","params",
				"price","units","notional","sl","tp",
				"equity_before","equity_after","pnl","pnl_pct","reason","loop",
			])
	return {"trades": trades_path}


def _params_summary(params: StrategyParams) -> Dict[str, float]:
	return {
		"fast": float(params.fast),
		"slow": float(params.slow),
		"macd_signal": float(params.macd_signal),
		"rsi_period": float(params.rsi_period),
		"rsi_low": float(params.rsi_low),
		"rsi_high": float(params.rsi_high),
		"bb_window": float(params.bb_window),
		"bb_mult": float(params.bb_mult),
		"atr_period": float(params.atr_period),
	}


def _update_params_from_opt(params: StrategyParams, strategy: str, best: Dict[str, float]) -> StrategyParams:
	new = StrategyParams(**asdict(params))
	if strategy == "sma":
		if "fast" in best: new.fast = int(best["fast"]) 
		if "slow" in best: new.slow = int(best["slow"]) 
	elif strategy == "rsi":
		if "rsi_period" in best: new.rsi_period = int(best["rsi_period"]) 
		if "rsi_low" in best: new.rsi_low = float(best["rsi_low"]) 
		if "rsi_high" in best: new.rsi_high = float(best["rsi_high"]) 
	elif strategy == "macd":
		if "fast" in best: new.fast = int(best["fast"]) 
		if "slow" in best: new.slow = int(best["slow"]) 
		if "macd_signal" in best: new.macd_signal = int(best["macd_signal"]) 
	elif strategy == "bollinger":
		if "bb_window" in best: new.bb_window = int(best["bb_window"]) 
		if "bb_mult" in best: new.bb_mult = float(best["bb_mult"]) 
	return new


def run_paper_trading(cfg: PaperConfig, base_params: StrategyParams) -> None:
	logs = _ensure_logs(cfg.log_dir)
	refresh = cfg.refresh_seconds or _interval_to_seconds(cfg.interval)
	bps_to_frac = 1.0 / 10000.0
	commission_frac = cfg.commission_bps * bps_to_frac
	spread_frac = cfg.spread_bps * bps_to_frac

	params_by_symbol: Dict[str, StrategyParams] = {s: StrategyParams(**asdict(base_params)) for s in cfg.symbols}
	equity: float = float(cfg.account_size)
	positions: Dict[str, PaperPosition] = {}
	used_notional: float = 0.0

	iteration = 0
	while True:
		iteration += 1
		for symbol in cfg.symbols:
			# Fetch latest data
			df = fetch_ohlcv(symbol=symbol, interval=cfg.interval, limit=cfg.limit)
			params = params_by_symbol[symbol]
			dfi = compute_indicators(df, params)
			last_close = float(dfi["close"].iloc[-1])
			last_high = float(dfi["high"].iloc[-1])
			last_low = float(dfi["low"].iloc[-1])
			last_atr = float(dfi.get("atr", pd.Series([np.nan])).iloc[-1])

			latest = evaluate_latest_signal(
				df=df, strategy=cfg.strategy,
				fast=params.fast, slow=params.slow, macd_signal=params.macd_signal,
				rsi_period=params.rsi_period, rsi_low=params.rsi_low, rsi_high=params.rsi_high,
				bb_window=params.bb_window, bb_mult=params.bb_mult,
			)

			# Manage position
			pos = positions.get(symbol)
			if pos is None and latest["signal"] == "buy":
				# Entry
				entry_price = last_close * (1 + spread_frac * 0.5)
				# Determine cap by available notional
				max_notional_allowed = max(equity * cfg.leverage - used_notional, 0.0)
				cap_notional = min(max_notional_allowed, equity * cfg.leverage * cfg.trade_pct)
				if cfg.size_mode == "atr_risk" and np.isfinite(last_atr) and last_atr > 0:
					sl, tp = project_sl_tp(entry_price, last_atr, cfg.sl_atr_mult, cfg.tp_atr_mult, direction="long")
					units = position_size_by_risk(equity, entry_price, sl, cfg.risk_pct)
					max_units = cap_notional / entry_price if entry_price > 0 else 0.0
					units = max(0.0, min(units, max_units))
				else:
					units = cap_notional / entry_price if entry_price > 0 else 0.0
					if np.isfinite(last_atr) and last_atr > 0:
						sl, tp = project_sl_tp(entry_price, last_atr, cfg.sl_atr_mult, cfg.tp_atr_mult, direction="long")
					else:
						sl, tp = (np.nan, np.nan)

				if units > 0:
					notional = units * entry_price
					fee = commission_frac * notional
					equity_before = equity
					equity -= fee
					used_notional += notional
					pos = PaperPosition(
						symbol=symbol, units=units, entry_price=entry_price, entry_time=_utcnow_iso(),
						sl=sl, tp=tp, notional=notional, strategy=cfg.strategy, params=_params_summary(params),
					)
					positions[symbol] = pos
					# Log entry
					with open(logs["trades"], "a", newline="") as f:
						writer = csv.writer(f)
						writer.writerow([
							_utcnow_iso(), symbol, "ENTRY", cfg.strategy, json.dumps(pos.params),
							f"{entry_price:.8f}", f"{units:.8f}", f"{notional:.2f}", f"{sl:.8f}", f"{tp:.8f}",
							f"{equity_before:.2f}", f"{equity:.2f}", "", "", latest.get("reason",""), iteration,
						])

			elif pos is not None:
				# Check exits: SL/TP or sell signal
				exit_reason = None
				if np.isfinite(pos.sl) and last_low <= pos.sl:
					exit_reason = "SL"
				elif np.isfinite(pos.tp) and last_high >= pos.tp:
					exit_reason = "TP"
				elif latest["signal"] == "sell":
					exit_reason = "SELL_SIGNAL"

				if exit_reason:
					exit_price = last_close * (1 - spread_frac * 0.5)
					notional_exit = pos.units * exit_price
					exit_fee = commission_frac * notional_exit
					pnl = pos.units * (exit_price - pos.entry_price) - exit_fee
					pnl_pct = pnl / max(equity, 1e-12)
					equity_before = equity
					equity += pnl
					used_notional = max(0.0, used_notional - pos.notional)
					del positions[symbol]
					with open(logs["trades"], "a", newline="") as f:
						writer = csv.writer(f)
						writer.writerow([
							_utcnow_iso(), symbol, "EXIT", cfg.strategy, json.dumps(pos.params),
							f"{exit_price:.8f}", f"{pos.units:.8f}", f"{notional_exit:.2f}", f"{pos.sl:.8f}", f"{pos.tp:.8f}",
							f"{equity_before:.2f}", f"{equity:.2f}", f"{pnl:.2f}", f"{pnl_pct:.6f}", exit_reason, iteration,
						])

		# Periodic optimization
		if cfg.optimize_every and iteration % cfg.optimize_every == 0:
			for symbol in cfg.symbols:
				df = fetch_ohlcv(symbol=symbol, interval=cfg.interval, limit=cfg.limit)
				res = optimize_strategy(
					df=df,
					strategy=cfg.strategy,
					n_trials=30,
					objective=cfg.objective,
					account_size=cfg.account_size,
					trade_pct=cfg.trade_pct,
					risk_pct=cfg.risk_pct,
					leverage=cfg.leverage,
					commission_bps=cfg.commission_bps,
					spread_bps=cfg.spread_bps,
					size_mode=cfg.size_mode,
				)
				params_by_symbol[symbol] = _update_params_from_opt(params_by_symbol[symbol], cfg.strategy, res.best_params)

		# Termination / wait
		if cfg.max_iterations is not None and iteration >= int(cfg.max_iterations):
			break
		time.sleep(refresh)