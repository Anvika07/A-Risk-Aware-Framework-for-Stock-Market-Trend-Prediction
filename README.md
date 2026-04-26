# A-Risk-Aware-Framework-for-Stock-Market-Trend-Prediction

This project provides a Python-based, risk-aware stock trend analyzer that focuses on short-term direction instead of exact price prediction.

It:

- Downloads historical market data with `yfinance`
- Computes technical indicators such as moving averages, RSI, MACD, ATR, and rolling volatility
- Classifies the next short-term move as `UP`, `DOWN`, or `STABLE`
- Estimates prediction confidence from the classifier and a Monte Carlo return band
- Produces a risk score and a volatility regime label
- Includes brief explainability and prediction failure analysis guidance

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run a text report:

```bash
python stock_trend_analyzer.py AAPL
```

Run JSON output:

```bash
python stock_trend_analyzer.py MSFT --json
```

Generate a chart:

```bash
python stock_trend_analyzer.py AAPL --plot --plot-output aapl.png
```

Optional arguments:

- `--period` controls the lookback window passed to `yfinance` such as `1y`, `2y`, or `5y`
- `--interval` controls the price interval such as `1d`

Example:

```bash
python stock_trend_analyzer.py NVDA --period 5y --interval 1d
```

## Web Frontend (HTML + Model API)

An interactive dashboard is included and inspired by the dark "quant terminal" reference style.

Start the web app:

```bash
python web_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Landing and terminal routes:

- `/` opens the login/landing screen
- `/terminal` opens the main analysis dashboard

Frontend files:

- `templates/login.html` for landing/login screen
- `templates/index.html` for structure
- `static/styles.css` for visual theme and responsive layout
- `static/app.js` for API calls and chart rendering

API routes:

- `GET /api/analyze?ticker=NVDA&period=2y&interval=1d`
- `GET /api/history?ticker=NVDA&period=2y&interval=1d`
- `GET /api/simulate?ticker=NVDA&period=2y&interval=1d&capital=10000`

The dashboard includes a Portfolio Simulation panel that is driven by model outputs
(predicted trend, confidence, risk score, and Monte Carlo band) and returns a
capital allocation, projected PnL, and stop-loss/take-profit guidance.

## Output Summary

The analyzer returns:

- `predicted_trend`: `UP`, `DOWN`, or `STABLE`
- `confidence_score`: model confidence in the direction class
- `backtest_accuracy_score`: walk-forward historical directional accuracy
- `backtest_summary`: evaluated sample count, windows, per-class recall, and confusion matrix
- `confidence_band`: Monte Carlo 10th/50th/90th percentile return range over the prediction horizon
- `risk_score`: 0 to 100 estimate of uncertainty and market unpredictability
- `volatility_classification`: `Stable`, `Risky`, or `Highly Volatile`
- `explanation`: short rationale using moving averages, RSI, MACD, price momentum, and volatility
- `failure_analysis_guidance`: common reasons a directional signal can fail

## Notes

- The system is intentionally directional and risk-aware. It does not predict exact future prices.
- Confidence and risk can diverge. A strong directional probability can still carry high volatility risk.
- `yfinance` requires network access when downloading live market history.
