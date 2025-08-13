import argparse
from typing import List

from btc_signal_engine.engine import StrategyParams
from btc_signal_engine.paper import PaperConfig, run_paper_trading


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Paper trading runner")
	p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Symbols to track")
	p.add_argument("--strategy", choices=["sma", "rsi", "macd", "bollinger"], default="macd")
	p.add_argument("--interval", default="1h")
	p.add_argument("--limit", type=int, default=500)
	p.add_argument("--account-size", dest="account_size", type=float, default=10000.0)
	p.add_argument("--trade-pct", dest="trade_pct", type=float, default=0.5)
	p.add_argument("--risk-pct", dest="risk_pct", type=float, default=0.01)
	p.add_argument("--leverage", type=float, default=1.0)
	p.add_argument("--commission-bps", dest="commission_bps", type=float, default=5.0)
	p.add_argument("--spread-bps", dest="spread_bps", type=float, default=10.0)
	p.add_argument("--size-mode", dest="size_mode", choices=["fixed_fraction", "atr_risk"], default="atr_risk")
	p.add_argument("--atr-period", dest="atr_period", type=int, default=14)
	p.add_argument("--sl-atr-mult", dest="sl_atr_mult", type=float, default=2.0)
	p.add_argument("--tp-atr-mult", dest="tp_atr_mult", type=float, default=4.0)
	p.add_argument("--optimize-every", dest="optimize_every", type=int, default=5, help="Optimize params every N loops")
	p.add_argument("--objective", choices=["sharpe_like", "total_return", "profit_factor"], default="sharpe_like")
	p.add_argument("--log-dir", dest="log_dir", default="/workspace/logs")
	p.add_argument("--refresh-seconds", dest="refresh_seconds", type=int, default=None, help="Override polling interval seconds (default from timeframe)")
	p.add_argument("--loops", type=int, default=0, help="Max loops (0 = run forever)")
	# base strategy params
	p.add_argument("--fast", type=int, default=12)
	p.add_argument("--slow", type=int, default=26)
	p.add_argument("--signal", dest="signal_window", type=int, default=9)
	p.add_argument("--rsi-period", dest="rsi_period", type=int, default=14)
	p.add_argument("--rsi-low", dest="rsi_low", type=float, default=30.0)
	p.add_argument("--rsi-high", dest="rsi_high", type=float, default=70.0)
	p.add_argument("--bb-window", dest="bb_window", type=int, default=20)
	p.add_argument("--bb-mult", dest="bb_mult", type=float, default=2.0)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	params = StrategyParams(
		fast=args.fast, slow=args.slow, macd_signal=args.signal_window,
		rsi_period=args.rsi_period, rsi_low=args.rsi_low, rsi_high=args.rsi_high,
		bb_window=args.bb_window, bb_mult=args.bb_mult, atr_period=args.atr_period,
	)
	cfg = PaperConfig(
		symbols=args.symbols,
		strategy=args.strategy,
		interval=args.interval,
		limit=args.limit,
		account_size=args.account_size,
		trade_pct=args.trade_pct,
		risk_pct=args.risk_pct,
		leverage=args.leverage,
		commission_bps=args.commission_bps,
		spread_bps=args.spread_bps,
		size_mode=args.size_mode,
		atr_period=args.atr_period,
		sl_atr_mult=args.sl_atr_mult,
		tp_atr_mult=args.tp_atr_mult,
		optimize_every=args.optimize_every,
		objective=args.objective,
		log_dir=args.log_dir,
		refresh_seconds=args.refresh_seconds,
		max_iterations=None if args.loops == 0 else int(args.loops),
	)
	run_paper_trading(cfg, params)


if __name__ == "__main__":
	main()