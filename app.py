from __future__ import annotations

import time
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from btc_signal_engine.data import fetch_ohlcv
from btc_signal_engine.engine import (
	StrategyParams,
	compute_indicators,
	evaluate_latest_signal,
	generate_signals,
	backtest_strategy,
)
from btc_signal_engine.risk import project_sl_tp
from btc_signal_engine.optimizer import optimize_strategy


st.set_page_config(page_title="Adaptive Trading Agent", layout="wide")

@st.cache_data(ttl=60)
def load_data(symbol: str, interval: str, limit: int) -> pd.DataFrame:
	return fetch_ohlcv(symbol=symbol, interval=interval, limit=limit)


def render_candles(df: pd.DataFrame, dfi: pd.DataFrame) -> go.Figure:
	fig = go.Figure()
	fig.add_trace(go.Candlestick(
		x=df.index,
		open=df["open"], high=df["high"], low=df["low"], close=df["close"],
		name="OHLC"
	))
	if "bb_upper" in dfi.columns:
		fig.add_trace(go.Scatter(x=df.index, y=dfi["bb_upper"], mode="lines", name="BB Upper"))
		fig.add_trace(go.Scatter(x=df.index, y=dfi["bb_mid"], mode="lines", name="BB Mid"))
		fig.add_trace(go.Scatter(x=df.index, y=dfi["bb_lower"], mode="lines", name="BB Lower"))
	if "sma_fast" in dfi.columns:
		fig.add_trace(go.Scatter(x=df.index, y=dfi["sma_fast"], mode="lines", name="SMA Fast"))
	if "sma_slow" in dfi.columns:
		fig.add_trace(go.Scatter(x=df.index, y=dfi["sma_slow"], mode="lines", name="SMA Slow"))
	fig.update_layout(height=500, xaxis_rangeslider_visible=False)
	return fig


def render_rsi(dfi: pd.DataFrame) -> go.Figure:
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=dfi.index, y=dfi["rsi"], mode="lines", name="RSI"))
	fig.update_layout(height=250)
	return fig


def render_macd(dfi: pd.DataFrame) -> go.Figure:
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=dfi.index, y=dfi["macd"], mode="lines", name="MACD"))
	fig.add_trace(go.Scatter(x=dfi.index, y=dfi["macd_signal"], mode="lines", name="Signal"))
	fig.add_trace(go.Bar(x=dfi.index, y=dfi["macd_hist"], name="Hist"))
	fig.update_layout(height=250)
	return fig


def render_equity(metrics: dict, title: str = "Backtest Metrics") -> None:
	st.markdown(f"**{title}**")
	cols = st.columns(3)
	cols[0].metric("Total Return", f"{metrics.get('total_return',0)*100:.2f}%")
	cols[1].metric("Win Rate", f"{metrics.get('win_rate',0)*100:.1f}%")
	cols[2].metric("Sharpe-like", f"{metrics.get('sharpe_like',0):.2f}")
	cols = st.columns(3)
	cols[0].metric("Profit Factor", f"{metrics.get('profit_factor',0):.2f}")
	cols[1].metric("Max Drawdown", f"{metrics.get('max_drawdown',0)*100:.2f}%")
	cols[2].metric("Avg Trade", f"{metrics.get('avg_trade_return',0)*100:.3f}%")


def main() -> None:
	st.sidebar.header("Market & Strategy")
	symbol = st.sidebar.selectbox("Symbol", [
		"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "BNBUSDT",
	])
	interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
	limit = st.sidebar.slider("Candles", 200, 1000, 500, step=50)

	strategy = st.sidebar.selectbox("Strategy", ["sma", "rsi", "macd", "bollinger"], index=2)
	with st.sidebar.expander("Parameters", expanded=False):
		fast = st.number_input("Fast", 5, 200, 20)
		slow = st.number_input("Slow", 10, 400, 50)
		macd_signal = st.number_input("MACD signal", 3, 30, 9)
		rsi_period = st.number_input("RSI period", 5, 50, 14)
		rsi_low = st.number_input("RSI low", 5, 50, 30)
		rsi_high = st.number_input("RSI high", 50, 95, 70)
		bb_window = st.number_input("BB window", 5, 60, 20)
		bb_mult = st.number_input("BB mult", 1.0, 4.0, 2.0, step=0.1)
		atr_period = st.number_input("ATR period", 5, 50, 14)
		sl_mult = st.number_input("SL ATR mult", 0.5, 10.0, 2.0, step=0.1)
		tp_mult = st.number_input("TP ATR mult", 0.5, 20.0, 4.0, step=0.1)

	st.sidebar.header("Risk & Costs")
	account_size = st.sidebar.number_input("Account size", 100.0, 10000000.0, 10000.0, step=100.0)
	leverage = st.sidebar.number_input("Leverage", 1.0, 100.0, 1.0, step=0.5)
	trade_pct = st.sidebar.slider("Trade fraction of equity", 0.0, 1.0, 1.0, step=0.05)
	risk_pct = st.sidebar.slider("Risk per trade", 0.0, 0.2, 0.01, step=0.005)
	size_mode = st.sidebar.selectbox("Sizing mode", ["fixed_fraction", "atr_risk"], index=0)
	commission_bps = st.sidebar.number_input("Commission (bps per side)", 0.0, 200.0, 5.0, step=0.5)
	spread_bps = st.sidebar.number_input("Spread (bps)", 0.0, 500.0, 10.0, step=0.5)

	st.sidebar.header("Live Tracking")
	refresh_s = st.sidebar.slider("Refresh seconds", 10, 300, 60, 5)
	live = st.sidebar.toggle("Live mode", value=False)

	# Load data
	df = load_data(symbol, interval, limit)
	params = StrategyParams(
		fast=int(fast), slow=int(slow), macd_signal=int(macd_signal),
		rsi_period=int(rsi_period), rsi_low=float(rsi_low), rsi_high=float(rsi_high),
		bb_window=int(bb_window), bb_mult=float(bb_mult), atr_period=int(atr_period)
	)
	dfi = compute_indicators(df, params)
	last_close = float(dfi["close"].iloc[-1])
	last_atr = float(dfi.get("atr", pd.Series([np.nan])).iloc[-1])
	sl, tp = project_sl_tp(last_close, last_atr, float(sl_mult), float(tp_mult), direction="long")

	# Latest signal
	latest = evaluate_latest_signal(
		df=df,
		strategy=strategy,
		fast=int(fast), slow=int(slow), macd_signal=int(macd_signal),
		rsi_period=int(rsi_period), rsi_low=float(rsi_low), rsi_high=float(rsi_high),
		bb_window=int(bb_window), bb_mult=float(bb_mult),
	)

	# Backtest snapshot for projected success rate
	metrics = backtest_strategy(
		df=df,
		strategy=strategy,
		fast=int(fast), slow=int(slow), macd_signal=int(macd_signal),
		rsi_period=int(rsi_period), rsi_low=float(rsi_low), rsi_high=float(rsi_high),
		bb_window=int(bb_window), bb_mult=float(bb_mult),
		account_size=float(account_size), trade_pct=float(trade_pct), risk_pct=float(risk_pct),
		leverage=float(leverage), commission_bps=float(commission_bps), spread_bps=float(spread_bps),
		atr_period=int(atr_period), sl_atr_mult=float(sl_mult), tp_atr_mult=float(tp_mult), size_mode=size_mode,
	)

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Signal", latest["signal"].upper())
	col2.metric("Projected SL", f"{sl:.2f}")
	col3.metric("Projected TP", f"{tp:.2f}")
	col4.metric("Proj. Win Rate", f"{metrics.get('win_rate',0)*100:.1f}%")

	# Charts selector
	charts = st.multiselect("Charts", ["Candles", "RSI", "MACD"], default=["Candles", "RSI", "MACD"])
	if "Candles" in charts:
		st.plotly_chart(render_candles(df, dfi), use_container_width=True)
	if "RSI" in charts:
		st.plotly_chart(render_rsi(dfi), use_container_width=True)
	if "MACD" in charts:
		st.plotly_chart(render_macd(dfi), use_container_width=True)

	# Optimization
	st.subheader("Self-learning optimizer")
	opt_cols = st.columns([1,1,1,2])
	objective = opt_cols[0].selectbox("Objective", ["sharpe_like", "total_return", "profit_factor"], index=0)
	n_trials = int(opt_cols[1].number_input("Trials", 10, 200, 30, step=10))
	start_opt = opt_cols[2].button("Run optimization")
	if start_opt:
		with st.spinner("Optimizing..."):
			res = optimize_strategy(
				df=df,
				strategy=strategy,
				n_trials=n_trials,
				objective=objective,
				account_size=float(account_size), trade_pct=float(trade_pct), risk_pct=float(risk_pct),
				leverage=float(leverage), commission_bps=float(commission_bps), spread_bps=float(spread_bps),
				size_mode=size_mode,
			)
			st.write("Best params:", res.best_params)
			render_equity(res.best_metrics, title="Optimized Backtest Metrics")

	if live:
		st.toast("Live mode enabled. Auto-refreshing.")
		st.experimental_rerun()


if __name__ == "__main__":
	main()