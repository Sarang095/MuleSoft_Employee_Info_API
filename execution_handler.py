from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import ccxt

import config
from logger import get_logger


@dataclass
class LivePosition:
    symbol: str
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    last_update_ts: float = field(default_factory=lambda: time.time())


class ExecutionHandler:
    def __init__(self, exchange: ccxt.Exchange) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.exchange = exchange
        self.position: Optional[LivePosition] = None

    def get_current_price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        price = float(ticker.get("last") or ticker.get("close") or 0.0)
        if price <= 0:
            raise RuntimeError(f"Invalid price fetched for {symbol}: {ticker}")
        return price

    def place_market_buy(self, symbol: str, quantity: float) -> dict:
        try:
            order = self.exchange.create_market_buy_order(symbol, quantity)
            self.logger.info(f"Placed market BUY: {order}")
            return order
        except Exception as exc:
            self.logger.error(f"Failed to place market BUY: {exc}")
            raise

    def place_market_sell(self, symbol: str, quantity: float) -> dict:
        try:
            order = self.exchange.create_market_sell_order(symbol, quantity)
            self.logger.info(f"Placed market SELL: {order}")
            return order
        except Exception as exc:
            self.logger.error(f"Failed to place market SELL: {exc}")
            raise

    def open_long(self, symbol: str, quantity: float, entry_price: float, stop_loss: float, take_profit: float) -> None:
        self.place_market_buy(symbol, quantity)
        self.position = LivePosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.logger.info(
            f"Opened long position: qty={quantity:.6f}, entry={entry_price:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}"
        )

    def close_position(self) -> None:
        if not self.position:
            return
        self.place_market_sell(self.position.symbol, self.position.quantity)
        self.logger.info("Closed position")
        self.position = None

    def update_trailing_stop(self, current_price: float, atr_value: float) -> None:
        if not self.position:
            return
        new_sl = current_price - (atr_value * config.ATR_MULTIPLIER)
        if new_sl > self.position.stop_loss:
            self.logger.info(f"Trailing SL updated from {self.position.stop_loss:.2f} to {new_sl:.2f}")
            self.position.stop_loss = new_sl
            self.position.last_update_ts = time.time()

    def evaluate_exit_conditions(self, current_price: float) -> Optional[str]:
        if not self.position:
            return None
        if current_price <= self.position.stop_loss:
            self.logger.info("Stop loss hit. Exiting position.")
            self.close_position()
            return "SL"
        if current_price >= self.position.take_profit:
            self.logger.info("Take profit hit. Exiting position.")
            self.close_position()
            return "TP"
        return None