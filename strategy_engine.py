from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import pandas as pd

import config
from logger import get_logger

Signal = Literal["BUY", "SELL", "HOLD"]
Regime = Literal["StrongBullTrend", "StrongBearTrend", "Ranging"]


@dataclass
class StrategyDecision:
    signal: Signal
    regime: Regime


class StrategyEngine:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def _determine_regime(self, row: pd.Series) -> Regime:
        ema50 = row["EMA50"]
        ema200 = row["EMA200"]
        adx = row["ADX14"]

        if adx > config.ADX_TREND_THRESHOLD and ema50 > ema200:
            return "StrongBullTrend"
        if adx > config.ADX_TREND_THRESHOLD and ema50 < ema200:
            return "StrongBearTrend"
        return "Ranging"

    def generate_signal(self, df: pd.DataFrame) -> StrategyDecision:
        if len(df) < 2:
            return StrategyDecision(signal="HOLD", regime="Ranging")

        row_prev = df.iloc[-2]
        row = df.iloc[-1]

        regime = self._determine_regime(row)

        close = row["close"]
        ema21 = row["EMA21"]
        rsi = row["RSI14"]
        rsi_prev = row_prev["RSI14"]
        macd = row["MACD"]
        macd_prev = row_prev["MACD"]
        macds = row["MACD_SIGNAL"]
        macds_prev = row_prev["MACD_SIGNAL"]
        bbu = row.get("BBU")
        bbl = row.get("BBL")
        bbm = row.get("BBM")

        # Default
        signal: Signal = "HOLD"

        if regime == "StrongBullTrend":
            # Buy: price near EMA21 AND RSI crossed up from 40-50 level
            near_ema = abs(close - ema21) / ema21 <= config.PRICE_NEAR_EMA_PCT
            rsi_cross_up = (config.RSI_BULL_CROSS_LOW <= rsi_prev < config.RSI_BULL_CROSS_HIGH) and (
                rsi >= config.RSI_BULL_CROSS_HIGH
            )
            if near_ema and rsi_cross_up:
                signal = "BUY"
            # Sell: bearish MACD crossover OR close below EMA21
            bearish_macd = (macd_prev > macds_prev) and (macd < macds)
            close_below_ema21 = close < ema21
            if bearish_macd or close_below_ema21:
                signal = "SELL"

        elif regime == "Ranging":
            # Buy: price <= LowerBB and RSI < 30
            if bbl is not None and close <= bbl and rsi < config.RSI_OVERSOLD:
                signal = "BUY"
            # Sell: price >= MiddleBB
            if bbm is not None and close >= bbm:
                signal = "SELL"

        elif regime == "StrongBearTrend":
            signal = "HOLD"

        return StrategyDecision(signal=signal, regime=regime)