import argparse
from typing import Optional

from btc_signal_engine.data import fetch_ohlcv, load_csv_ohlcv
from btc_signal_engine.engine import evaluate_latest_signal, backtest_strategy, compute_indicators, StrategyParams
from btc_signal_engine.risk import project_sl_tp
import numpy as np


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="BTC Trading Signal Engine")
	parser.add_argument("--strategy", required=True, choices=["sma", "rsi", "macd", "bollinger"], help="Strategy to run")
	parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (Binance format), e.g., BTCUSDT")
	parser.add_argument("--interval", default="1h", help="Interval, e.g., 1m, 5m, 15m, 1h, 4h, 1d")
	parser.add_argument("--limit", type=int, default=500, help="Number of candles to fetch (max 1000)")
	parser.add_argument("--data-csv", dest="data_csv", default=None, help="Path to local OHLCV CSV file to use instead of fetching")
	# strategy params
	parser.add_argument("--fast", type=int, default=20, help="Fast window (SMA, MACD fast EMA)")
	parser.add_argument("--slow", type=int, default=50, help="Slow window (SMA, MACD slow EMA)")
	parser.add_argument("--signal", dest="signal_window", type=int, default=9, help="MACD signal EMA window")
	parser.add_argument("--rsi-period", dest="rsi_period", type=int, default=14, help="RSI lookback period")
	parser.add_argument("--rsi-low", dest="rsi_low", type=float, default=30.0, help="RSI lower threshold")
	parser.add_argument("--rsi-high", dest="rsi_high", type=float, default=70.0, help="RSI upper threshold")
	parser.add_argument("--bb-window", dest="bb_window", type=int, default=20, help="Bollinger Bands MA window")
	parser.add_argument("--bb-mult", dest="bb_mult", type=float, default=2.0, help="Bollinger Bands stddev multiplier")
	parser.add_argument("--atr-period", dest="atr_period", type=int, default=14, help="ATR period")
	parser.add_argument("--sl-atr-mult", dest="sl_atr_mult", type=float, default=2.0, help="Stop-loss ATR multiplier")
	parser.add_argument("--tp-atr-mult", dest="tp_atr_mult", type=float, default=4.0, help="Take-profit ATR multiplier")
	# risk/cost params
	parser.add_argument("--account-size", dest="account_size", type=float, default=10000.0, help="Account size in quote currency")
	parser.add_argument("--trade-pct", dest="trade_pct", type=float, default=1.0, help="Fraction of equity allocated per trade (0-1)")
	parser.add_argument("--risk-pct", dest="risk_pct", type=float, default=0.01, help="Risk per trade as fraction of equity (0-1)")
	parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiple")
	parser.add_argument("--commission-bps", dest="commission_bps", type=float, default=0.0, help="Commission in basis points per side")
	parser.add_argument("--spread-bps", dest="spread_bps", type=float, default=0.0, help="Effective spread in basis points")
	parser.add_argument("--size-mode", dest="size_mode", choices=["fixed_fraction", "atr_risk"], default="fixed_fraction", help="Position sizing mode")
	parser.add_argument("--backtest", action="store_true", help="Run a backtest and print metrics")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.data_csv:
		df = load_csv_ohlcv(args.data_csv)
	else:
		df = fetch_ohlcv(symbol=args.symbol, interval=args.interval, limit=args.limit)

	latest = evaluate_latest_signal(
		df=df,
		strategy=args.strategy,
		fast=args.fast,
		slow=args.slow,
		macd_signal=args.signal_window,
		rsi_period=args.rsi_period,
		rsi_low=args.rsi_low,
		rsi_high=args.rsi_high,
		bb_window=args.bb_window,
		bb_mult=args.bb_mult,
	)

	# Compute indicators for SL/TP projection
	params = StrategyParams(
		fast=args.fast,
		slow=args.slow,
		macd_signal=args.signal_window,
		rsi_period=args.rsi_period,
		rsi_low=args.rsi_low,
		rsi_high=args.rsi_high,
		bb_window=args.bb_window,
		bb_mult=args.bb_mult,
		atr_period=args.atr_period,
	)
	dfi = compute_indicators(df, params)
	last_close = float(dfi["close"].iloc[-1])
	last_atr = float(dfi["atr"].iloc[-1]) if "atr" in dfi.columns else float("nan")
	sl, tp = project_sl_tp(last_close, last_atr, args.sl_atr_mult, args.tp_atr_mult, direction="long")

	print(f"Symbol: {args.symbol if not args.data_csv else 'CSV'} | Interval: {args.interval if not args.data_csv else 'N/A'} | Candles: {len(df)}")
	print(f"Last candle: {df.index[-1]} Close: {last_close:.2f}")
	print(f"Strategy: {args.strategy.upper()} -> Signal: {latest['signal'].upper()} | Reason: {latest['reason']}")
	if np.isfinite(last_atr):
		print(f"ATR({args.atr_period}): {last_atr:.2f} | Projected SL: {sl:.2f} | TP: {tp:.2f}")

	if args.backtest:
		metrics = backtest_strategy(
			df=df,
			strategy=args.strategy,
			fast=args.fast,
			slow=args.slow,
			macd_signal=args.signal_window,
			rsi_period=args.rsi_period,
			rsi_low=args.rsi_low,
			rsi_high=args.rsi_high,
			bb_window=args.bb_window,
			bb_mult=args.bb_mult,
			account_size=args.account_size,
			trade_pct=args.trade_pct,
			risk_pct=args.risk_pct,
			leverage=args.leverage,
			commission_bps=args.commission_bps,
			spread_bps=args.spread_bps,
			atr_period=args.atr_period,
			sl_atr_mult=args.sl_atr_mult,
			tp_atr_mult=args.tp_atr_mult,
			size_mode=args.size_mode,
		)
		print("Backtest metrics:")
		for key, value in metrics.items():
			if isinstance(value, float):
				print(f"  {key}: {value:.4f}")
			else:
				print(f"  {key}: {value}")


if __name__ == "__main__":
	main()