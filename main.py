import argparse
from typing import Optional

from btc_signal_engine.data import fetch_ohlcv, load_csv_ohlcv
from btc_signal_engine.engine import evaluate_latest_signal, backtest_strategy


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="BTC Trading Signal Engine")
	parser.add_argument("--strategy", required=True, choices=["sma", "rsi", "macd", "bollinger"], help="Strategy to run")
	parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (Binance format), e.g., BTCUSDT")
	parser.add_argument("--interval", default="1h", help="Binance interval, e.g., 1m, 5m, 15m, 1h, 4h, 1d")
	parser.add_argument("--limit", type=int, default=500, help="Number of candles to fetch (max 1000)")
	parser.add_argument("--data-csv", dest="data_csv", default=None, help="Path to local OHLCV CSV file to use instead of fetching")
	parser.add_argument("--fast", type=int, default=20, help="Fast window (SMA, MACD fast EMA)")
	parser.add_argument("--slow", type=int, default=50, help="Slow window (SMA, MACD slow EMA)")
	parser.add_argument("--signal", dest="signal_window", type=int, default=9, help="MACD signal EMA window")
	parser.add_argument("--rsi-period", dest="rsi_period", type=int, default=14, help="RSI lookback period")
	parser.add_argument("--rsi-low", dest="rsi_low", type=float, default=30.0, help="RSI lower threshold")
	parser.add_argument("--rsi-high", dest="rsi_high", type=float, default=70.0, help="RSI upper threshold")
	parser.add_argument("--bb-window", dest="bb_window", type=int, default=20, help="Bollinger Bands moving average window")
	parser.add_argument("--bb-mult", dest="bb_mult", type=float, default=2.0, help="Bollinger Bands standard deviation multiplier")
	parser.add_argument("--backtest", action="store_true", help="Run a simple backtest and print metrics")
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

	print(f"Symbol: {args.symbol if not args.data_csv else 'CSV'} | Interval: {args.interval if not args.data_csv else 'N/A'} | Candles: {len(df)}")
	print(f"Last candle: {df.index[-1]} Close: {df['close'].iloc[-1]:.2f}")
	print(f"Strategy: {args.strategy.upper()} -> Signal: {latest['signal'].upper()} | Reason: {latest['reason']}")

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
		)
		print("Backtest metrics:")
		for key, value in metrics.items():
			if isinstance(value, float):
				print(f"  {key}: {value:.4f}")
			else:
				print(f"  {key}: {value}")


if __name__ == "__main__":
	main()