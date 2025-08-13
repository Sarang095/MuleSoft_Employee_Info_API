from __future__ import annotations

from typing import Optional
import io
import requests
import pandas as pd


BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_FALLBACK_URLS = [
	"https://api1.binance.com",
	"https://api2.binance.com",
	"https://api3.binance.com",
	"https://api.binance.us",
]


def fetch_ohlcv(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
	"""Fetch OHLCV candles from public sources with fallbacks (Binance -> Coinbase -> Kraken).

	Args:
		symbol: Trading pair symbol, e.g., "BTCUSDT" (Binance-style). For fallbacks, BTC symbols are supported out of the box.
		interval: Kline interval, e.g., "1m", "5m", "15m", "1h", "4h", "1d".
		limit: Number of candles to fetch (1-1000 typical). Some providers cap the last N (e.g., 300 on Coinbase).

	Returns:
		DataFrame indexed by UTC datetime with columns: open, high, low, close, volume.
	"""
	params = {"symbol": symbol.upper(), "interval": interval, "limit": min(max(int(limit), 1), 1000)}
	# Try Binance and known fallbacks first
	last_err: Optional[Exception] = None
	for base in [BINANCE_BASE_URL] + BINANCE_FALLBACK_URLS:
		try:
			url = f"{base}/api/v3/klines"
			resp = requests.get(url, params=params, timeout=20)
			resp.raise_for_status()
			raw = resp.json()
			cols = [
				"open_time","open","high","low","close","volume",
				"close_time","quote_asset_volume","num_trades",
				"taker_buy_base","taker_buy_quote","ignore",
			]
			df = pd.DataFrame(raw, columns=cols)
			for col in ["open","high","low","close","volume"]:
				df[col] = df[col].astype(float)
			df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
			df = df.set_index("timestamp")[ ["open","high","low","close","volume"] ]
			return df
		except Exception as e:
			last_err = e

	# Coinbase fallback (BTC only by default)
	try:
		product_id, granularity = _map_to_coinbase(symbol, interval)
		if product_id is not None and granularity is not None:
			url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
			resp = requests.get(url, params={"granularity": granularity}, timeout=20)
			resp.raise_for_status()
			# Coinbase returns arrays: [time, low, high, open, close, volume] newest first
			raw = resp.json()
			df = pd.DataFrame(raw, columns=["time","low","high","open","close","volume"]).astype({
				"open": float, "high": float, "low": float, "close": float, "volume": float,
			})
			df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
			df = df.sort_values("timestamp").set_index("timestamp")[ ["open","high","low","close","volume"] ]
			if limit:
				df = df.tail(int(limit))
			return df
	except Exception as e:
		last_err = e

	# Kraken fallback (BTC/USD by default)
	try:
		pair, interval_minutes = _map_to_kraken(symbol, interval)
		if pair is not None and interval_minutes is not None:
			url = "https://api.kraken.com/0/public/OHLC"
			resp = requests.get(url, params={"pair": pair, "interval": interval_minutes}, timeout=20)
			resp.raise_for_status()
			raw = resp.json()
			result_key = next((k for k in raw.get("result", {}) if k != "last"), None)
			if not result_key:
				raise ValueError("Kraken response missing result key")
			rows = raw["result"][result_key]
			df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","count"]).astype({
				"open": float, "high": float, "low": float, "close": float, "volume": float,
			})
			df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
			df = df.set_index("timestamp").sort_index()[ ["open","high","low","close","volume"] ]
			if limit:
				df = df.tail(int(limit))
			return df
	except Exception as e:
		last_err = e

	raise RuntimeError(f"Failed to fetch OHLCV from all providers: {last_err}")


def load_csv_ohlcv(path: str) -> pd.DataFrame:
	"""Load OHLCV candles from a CSV file.

	The CSV must have at least these columns: timestamp, open, high, low, close, volume.
	The timestamp will be parsed to UTC and used as the index.
	"""
	df = pd.read_csv(path)
	required = {"timestamp", "open", "high", "low", "close", "volume"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
	df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
	df = df.set_index("timestamp")[ ["open","high","low","close","volume"] ]
	for col in ["open","high","low","close","volume"]:
		df[col] = df[col].astype(float)
	return df


def _map_to_coinbase(symbol: str, interval: str) -> tuple[Optional[str], Optional[int]]:
	"""Map Binance-like inputs to Coinbase product and granularity seconds."""
	interval_map = {"1m":60, "5m":300, "15m":900, "1h":3600, "4h":21600, "1d":86400}
	gran = interval_map.get(interval)
	if gran is None:
		return None, None
	sym = symbol.upper()
	# Minimal mapping for BTC pairs
	if sym in {"BTCUSDT", "BTCUSD", "XBTUSD"}:
		return "BTC-USD", gran
	# Generic hyphenation: e.g., ETHUSDT -> ETH-USDT (may or may not exist)
	if sym.endswith("USDT"):
		return f"{sym[:-4]}-USDT", gran
	if sym.endswith("USD"):
		return f"{sym[:-3]}-USD", gran
	return None, None


def _map_to_kraken(symbol: str, interval: str) -> tuple[Optional[str], Optional[int]]:
	"""Map Binance-like inputs to Kraken pair and interval minutes."""
	interval_map = {"1m":1, "5m":5, "15m":15, "1h":60, "4h":240, "1d":1440}
	mins = interval_map.get(interval)
	if mins is None:
		return None, None
	sym = symbol.upper()
	# Minimal mapping for BTC pairs
	if sym in {"BTCUSDT", "BTCUSD", "XBTUSD"}:
		return "XBTUSD", mins
	return None, None