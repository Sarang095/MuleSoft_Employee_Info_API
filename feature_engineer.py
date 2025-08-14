from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np

# Workaround for pandas_ta importing `NaN` from numpy
if not hasattr(np, "NaN"):
	setattr(np, "NaN", np.nan)

import pandas_ta as ta

import config
from logger import get_logger


class FeatureEngineer:
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame missing required OHLCV columns: {required_cols}")

        out = df.copy()

        # EMAs
        out["EMA21"] = ta.ema(out["close"], length=config.EMA_FAST)
        out["EMA50"] = ta.ema(out["close"], length=config.EMA_MED)
        out["EMA200"] = ta.ema(out["close"], length=config.EMA_SLOW)

        # ADX
        adx_df = ta.adx(high=out["high"], low=out["low"], close=out["close"], length=config.ADX_PERIOD)
        out["ADX14"] = adx_df[adx_df.columns[0]]  # usually 'ADX_14'

        # RSI
        out["RSI14"] = ta.rsi(out["close"], length=config.RSI_PERIOD)

        # Bollinger Bands
        bb = ta.bbands(close=out["close"], length=config.BB_PERIOD, std=config.BB_STDDEV)
        out["BBL"] = bb[bb.columns[0]]  # lower band
        out["BBM"] = bb[bb.columns[1]]  # middle band (SMA)
        out["BBU"] = bb[bb.columns[2]]  # upper band

        # MACD
        macd = ta.macd(
            close=out["close"], fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL
        )
        out["MACD"] = macd[macd.columns[0]]
        out["MACD_SIGNAL"] = macd[macd.columns[1]]
        out["MACD_HIST"] = macd[macd.columns[2]]

        # ATR
        atr = ta.atr(high=out["high"], low=out["low"], close=out["close"], length=config.ATR_PERIOD)
        out["ATR14"] = atr

        out.dropna(inplace=True)
        return out