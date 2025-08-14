from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Generator, Optional

import ccxt
import numpy as np
import pandas as pd

from logger import get_logger
import config


class DataHandler:
	def __init__(
		self,
		exchange_id: str = config.EXCHANGE_ID,
		api_key: str = config.API_KEY,
		api_secret: str = config.API_SECRET,
		sandbox: bool = config.SANDBOX_MODE,
	) -> None:
		self.logger = get_logger(self.__class__.__name__)
		self.exchange = self._init_exchange_with_fallbacks(exchange_id, api_key, api_secret, sandbox)

	def _build_exchange(self, exchange_id: str, api_key: str, api_secret: str, sandbox: bool) -> ccxt.Exchange:
		try:
			exchange_class = getattr(ccxt, exchange_id)
		except AttributeError as exc:
			raise ValueError(f"Unsupported exchange '{exchange_id}'") from exc

		ex = exchange_class({
			"apiKey": api_key,
			"secret": api_secret,
			"enableRateLimit": True,
		})
		if sandbox and hasattr(ex, "set_sandbox_mode"):
			try:
				ex.set_sandbox_mode(True)
				self.logger.info(f"Sandbox mode enabled for {exchange_id}")
			except Exception as exc:
				self.logger.warning(f"Failed to enable sandbox for {exchange_id}: {exc}")
		return ex

	def _init_exchange_with_fallbacks(self, exchange_id: str, api_key: str, api_secret: str, sandbox: bool) -> ccxt.Exchange:
		candidates = [exchange_id] + [ex for ex in config.EXCHANGE_FALLBACKS if ex != exchange_id]
		last_exc: Optional[Exception] = None
		for ex_id in candidates:
			try:
				ex = self._build_exchange(ex_id, api_key, api_secret, sandbox)
				ex.load_markets()
				self.logger.info(f"Connected to exchange: {ex_id}")
				return ex
			except Exception as exc:
				last_exc = exc
				self.logger.error(f"Failed to init/load markets for {ex_id}: {exc}")
		raise last_exc if last_exc else RuntimeError("Failed to initialize any exchange")

	def _parse_since(self, since: Optional[str | int | float]) -> Optional[int]:
		if since is None:
			return None
		if isinstance(since, (int, float)):
			return int(since)
		try:
			dt = pd.to_datetime(since, utc=True)
			return int(dt.timestamp() * 1000)
		except Exception:
			self.logger.warning(f"Could not parse 'since': {since}")
			return None

	def fetch_ohlcv(
		self,
		symbol: str,
		timeframe: str = "1d",
		since: Optional[str | int | float] = None,
		limit: int = 1000,
		until: Optional[str | int | float] = None,
		max_bars: Optional[int] = None,
	) -> pd.DataFrame:
		"""
		Fetch historical OHLCV data with pagination and return as a cleaned DataFrame.
		"""
		since_ms = self._parse_since(since)
		until_ms = self._parse_since(until)

		all_ohlcv: list[list[float]] = []
		fetch_since = since_ms

		# Determine timeframe in seconds
		tf_seconds = self.exchange.parse_timeframe(timeframe)
		ms_per_candle = tf_seconds * 1000

		self.logger.info(
			f"Fetching OHLCV for {symbol} {timeframe} from {since} to {until} with limit {limit}"
		)

		try:
			while True:
				ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
				if not ohlcv:
					break

				all_ohlcv.extend(ohlcv)

				last_ts = ohlcv[-1][0]
				next_since = last_ts + ms_per_candle

				if until_ms is not None and next_since >= until_ms:
					break

				if max_bars is not None and len(all_ohlcv) >= max_bars:
					all_ohlcv = all_ohlcv[:max_bars]
					break

				# Prevent tight loop
				fetch_since = next_since
				time.sleep(self.exchange.rateLimit / 1000.0)
		except ccxt.BaseError as exc:
			self.logger.error(f"Error fetching OHLCV: {exc}")
			raise

		if len(all_ohlcv) == 0:
			raise RuntimeError("No OHLCV data returned.")

		df = pd.DataFrame(
			all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
		)
		df.drop_duplicates(subset=["timestamp"], inplace=True)
		df.sort_values("timestamp", inplace=True)
		df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
		df.set_index("datetime", inplace=True)
		df = df[["open", "high", "low", "close", "volume"]]
		df = df.astype(float)

		# Basic cleaning: forward fill then drop remaining NaNs
		df.replace([np.inf, -np.inf], np.nan, inplace=True)
		df.ffill(inplace=True)
		df.dropna(inplace=True)

		self.logger.info(f"Fetched {len(df)} rows of OHLCV for {symbol} {timeframe}")
		return df

	def stream_realtime_ohlcv(
		self,
		symbol: str,
		timeframe: str = "1m",
		lookback: int = 500,
	) -> Generator[pd.DataFrame, None, None]:
		"""
		Poll the exchange periodically and yield the latest OHLCV DataFrame snapshot.
		"""
		last_ts: Optional[pd.Timestamp] = None
		while True:
			try:
				df = self.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback)
				if last_ts is None or df.index[-1] > last_ts:
					last_ts = df.index[-1]
					yield df
			except Exception as exc:
				self.logger.error(f"Realtime stream error: {exc}")
				time.sleep(5)
				continue
			time.sleep(config.LOOP_INTERVAL_SEC)