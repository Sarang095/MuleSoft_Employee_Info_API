from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

import config
from data_handler import DataHandler
from feature_engineer import FeatureEngineer
from strategy_engine import StrategyEngine
from risk_manager import RiskManager
from execution_handler import ExecutionHandler
from backtest import run_in_sample_out_of_sample
from logger import get_logger


def run_backtests():
    logger = get_logger("MainBacktest")
    train, test = run_in_sample_out_of_sample()

    def print_metrics(tag: str, m: dict):
        logger.info(f"==== {tag} ====")
        for k, v in m.items():
            logger.info(f"{k}: {v}")

    print_metrics("IN-SAMPLE (TRAIN)", train.metrics)
    print_metrics("OUT-OF-SAMPLE (TEST)", test.metrics)


def run_live():
    logger = get_logger("MainLive")
    dh = DataHandler()
    fe = FeatureEngineer()
    engine = StrategyEngine()
    risk = RiskManager()
    ex = ExecutionHandler(dh.exchange)

    logger.info("Starting live trading loop")
    for df in dh.stream_realtime_ohlcv(config.SYMBOL, timeframe=config.TIMEFRAME, lookback=600):
        try:
            df_features = fe.add_features(df)
            decision = engine.generate_signal(df_features)
            last = df_features.iloc[-1]
            atr = float(last["ATR14"])
            price = float(last["close"])

            if ex.position is None:
                if decision.signal == "BUY":
                    equity = float(dh.exchange.fetch_balance().get("USDT", {}).get("free", 0.0))
                    if equity <= 0:
                        logger.warning("No USDT balance available to trade")
                        continue
                    order_spec = risk.build_order(config.SYMBOL, price, atr, equity)
                    # RiskManager returns position_size in base units (BTC) given we divided by ATR*multiplier in $ terms,
                    # but to be robust, if exchange requires base quantity, use it directly.
                    qty = order_spec.quantity
                    ex.open_long(config.SYMBOL, qty, price, order_spec.stop_loss, order_spec.take_profit)
            else:
                # Update trailing SL
                ex.update_trailing_stop(price, atr)
                # Evaluate exits
                ex.evaluate_exit_conditions(price)
        except Exception as exc:
            logger.error(f"Live loop error: {exc}")
            time.sleep(5)
            continue


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Hybrid Regime-Adaptive Crypto Trading Bot")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.mode == "backtest":
        run_backtests()
    else:
        run_live()