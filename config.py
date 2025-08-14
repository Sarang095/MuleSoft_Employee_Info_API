import os
from datetime import datetime

# Exchange and API configuration
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
SANDBOX_MODE = os.getenv("SANDBOX_MODE", "false").lower() == "true"
# Fallback exchanges if the primary is unavailable (regional restrictions, outages)
EXCHANGE_FALLBACKS = [ex.strip() for ex in os.getenv(
	"EXCHANGE_FALLBACKS", "kucoin,bybit,okx,binanceus"
).split(",") if ex.strip()]

# Trading configuration
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1d")  # For backtesting; live can use shorter like '1m'

# Backtest date ranges (YYYY-MM-DD)
BACKTEST_START = os.getenv("BACKTEST_START", "2019-01-01")
BACKTEST_END = os.getenv("BACKTEST_END", datetime.utcnow().strftime("%Y-%m-%d"))

# In-sample / Out-of-sample ranges (optional)
TRAIN_START = os.getenv("TRAIN_START", "2019-01-01")
TRAIN_END = os.getenv("TRAIN_END", "2022-12-31")
TEST_START = os.getenv("TEST_START", "2023-01-01")
TEST_END = os.getenv("TEST_END", datetime.utcnow().strftime("%Y-%m-%d"))

# Indicator parameters
EMA_FAST = int(os.getenv("EMA_FAST", 21))
EMA_MED = int(os.getenv("EMA_MED", 50))
EMA_SLOW = int(os.getenv("EMA_SLOW", 200))

ADX_PERIOD = int(os.getenv("ADX_PERIOD", 14))
ADX_TREND_THRESHOLD = float(os.getenv("ADX_TREND_THRESHOLD", 25.0))

RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", 30.0))
RSI_BULL_CROSS_LOW = float(os.getenv("RSI_BULL_CROSS_LOW", 40.0))
RSI_BULL_CROSS_HIGH = float(os.getenv("RSI_BULL_CROSS_HIGH", 50.0))

BB_PERIOD = int(os.getenv("BB_PERIOD", 20))
BB_STDDEV = float(os.getenv("BB_STDDEV", 2.0))

# MACD parameters (interpreting "14-period MACD" as fast=14, slow=28, signal=9)
MACD_FAST = int(os.getenv("MACD_FAST", 14))
MACD_SLOW = int(os.getenv("MACD_SLOW", 28))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 9))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", 2.0))

# Risk management
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))  # 1% per trade
RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", 2.0))
PRICE_NEAR_EMA_PCT = float(os.getenv("PRICE_NEAR_EMA_PCT", 0.005))  # 0.5%

# Costs and slippage
TRADING_FEE = float(os.getenv("TRADING_FEE", 0.001))  # 0.1% per trade
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", 0.0005))  # 0.05%

# Live trading polling interval (seconds)
LOOP_INTERVAL_SEC = int(os.getenv("LOOP_INTERVAL_SEC", 15))

# Plotting
PLOT_OUTPUT = os.getenv("PLOT_OUTPUT", "plots")

# Logging
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")