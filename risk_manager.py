from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import config


@dataclass
class OrderSpec:
    symbol: str
    action: str  # 'BUY'
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float


class RiskManager:
    def __init__(self) -> None:
        self.risk_per_trade = config.RISK_PER_TRADE
        self.atr_multiplier = config.ATR_MULTIPLIER
        self.rrr = config.RISK_REWARD_RATIO

    def build_order(self, symbol: str, entry_price: float, atr_value: float, total_equity: float) -> OrderSpec:
        if atr_value <= 0 or entry_price <= 0 or total_equity <= 0:
            raise ValueError("Invalid inputs for risk management calculations")

        risk_amount = total_equity * self.risk_per_trade
        position_size = risk_amount / (atr_value * self.atr_multiplier)

        stop_loss = entry_price - (atr_value * self.atr_multiplier)
        take_profit = entry_price + self.rrr * (entry_price - stop_loss)

        return OrderSpec(
            symbol=symbol,
            action="BUY",
            quantity=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )