## Hybrid Regime-Adaptive Cryptocurrency Trading Bot

This project implements a modular, production-grade crypto trading bot designed to classify market regimes and deploy the best-suited sub-strategy for BTC/USDT spot trading. The system supports both backtesting (event-driven via Backtrader) and live trading (via ccxt), with volatility-adjusted risk management, realistic costs, and robust logging.

### How the platform works
- **DataHandler**: Fetches historical OHLCV data and streams real-time data via ccxt.
- **FeatureEngineer**: Computes EMA21/50/200, ADX(14), RSI(14), Bollinger Bands(20,2), MACD(14,28,9), ATR(14).
- **StrategyEngine**: Classifies regime and generates BUY/SELL/HOLD signals:
  - StrongBullTrend: EMA50 > EMA200 and ADX > threshold. Buy near EMA21 with RSI cross-up from 40-50; sell on bearish MACD cross or close < EMA21.
  - Ranging: ADX < threshold. Buy at lower BB with RSI < 30; sell at middle BB.
  - StrongBearTrend: Hold (no new longs).
- **RiskManager**: Positions sized by volatility. Stop-loss and take-profit computed from ATR.
- **ExecutionHandler**: Places orders and manages ATR trailing stop.
- **Backtesting**: Evaluates performance with fees and slippage; supports in-sample/out-of-sample.
- **Logging**: Rotating logs to `logs/trading_bot.log`.

### Requirements
- Python 3.11+
- See `requirements.txt` for Python packages.

### Setup
1. Create and activate a virtual environment (recommended).
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration
Set parameters and API keys in `config.py`, or via environment variables. Key settings:
- `SYMBOL`, `TIMEFRAME`
- Indicator parameters: `EMA_*`, `ADX_*`, `RSI_*`, `BB_*`, `MACD_*`, `ATR_*`
- Risk: `RISK_PER_TRADE`, `ATR_MULTIPLIER`, `RISK_REWARD_RATIO`, `PRICE_NEAR_EMA_PCT`
- Costs: `TRADING_FEE`, `SLIPPAGE_PCT`
- Backtest ranges: `TRAIN_*`, `TEST_*`
- API: `EXCHANGE_ID`, `API_KEY`, `API_SECRET`, `SANDBOX_MODE`

### Run Backtests
By default, runs in-sample on `TRAIN_*` and out-of-sample on `TEST_*` ranges. Plots saved in `plots/`.
```bash
python main.py --mode backtest
```
It prints metrics:
- Starting/Final Value, Net Profit, Annualized Return %, Sharpe Ratio, Max Drawdown %, Total Trades, Win Rate %, Profit Factor

### Live Trading
Use at your own risk. Ensure credentials and balances are correct. Consider `SANDBOX_MODE=true` when supported.
```bash
export API_KEY=your_key
export API_SECRET=your_secret
export SANDBOX_MODE=true
python main.py --mode live
```
The bot will:
- Fetch recent OHLCV snapshots in a loop
- Compute indicators and signals
- Place buy orders on BUY signals and manage trailing stop/TP

### Notes
- Fees and slippage are included in backtests.
- The code uses pandas-ta for indicators. If TA-Lib is preferred, swap implementations accordingly.
- For portfolio value plots, see `plots/equity_train.png` and `plots/equity_test.png`.

### Safety & Disclaimer
This software is for educational purposes. Cryptocurrency trading carries significant risk. No warranties are provided. Use sandbox/testnet and proceed carefully.