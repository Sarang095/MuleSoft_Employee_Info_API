# BTC Trading Signal Engine

A lightweight Python engine that generates trading signals for BTC using common technical strategies (SMA crossover, RSI, MACD, Bollinger Bands). Fetches OHLCV data from Binance or from a local CSV.

## Quick start

1) Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run once to get a signal:

```bash
python main.py --strategy sma --symbol BTCUSDT --interval 1h --limit 500
```

3) Backtest a strategy:

```bash
python main.py --strategy macd --symbol BTCUSDT --interval 1h --limit 1000 --backtest
```

4) Use local CSV data instead of fetching (must contain columns: timestamp/open/high/low/close/volume):

```bash
python main.py --strategy rsi --data-csv path/to/btc_ohlcv.csv
```

## Strategies
- sma: Fast/slow simple moving average crossover
- rsi: RSI overbought/oversold reversals
- macd: MACD line and signal line crossovers
- bollinger: Mean reversion around Bollinger Bands

## Disclaimer
This software is for informational and educational purposes only and does not constitute financial advice. Use at your own risk.