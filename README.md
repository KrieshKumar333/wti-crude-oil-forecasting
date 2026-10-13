# WTI Crude Oil Price Forecasting Model

Built this to understand how quantitative macro research works in practice -
specifically, how you systematically combine economic signals to forecast commodity prices.

Crude oil sits at the intersection of macroeconomics, geopolitics, supply chains,
and financial markets. Every Wednesday at 10:30 AM EST, the EIA releases weekly
US petroleum inventory data and the market moves on it. I wanted to know:
can you systematically predict oil direction using the macro signals that commodity
desks watch? This project is my answer.

## What I Built

A 4-week forward return forecasting model for WTI crude oil:

- **23 features** engineered across 6 categories:
  - **Momentum**: 1W, 4W, 13W returns; 4W/13W realised volatility; Brent-WTI spread; 52W z-score
  - **Inventory**: EIA weekly crude levels, inventory surprise vs 5-year seasonal average, days-of-supply
  - **Dollar (DXY)**: 4W return, 4W volatility - oil is priced in USD, inverse relationship
  - **Risk Sentiment**: VIX level, VIX 4W change, S&P 500 4W return
  - **Interest Rates**: US 10Y Treasury yield, 10Y-2Y yield curve spread (recession signal)
  - **Seasonal**: sin/cos month encoding, Q1/Q3 dummies (heating/driving seasons)
- **XGBoost model** - handles non-linear macro relationships that linear regression misses
- **Walk-forward expanding-window backtest** - the only valid methodology for time-series:
  for each test week, train on ALL prior data, predict 4 weeks ahead. Zero look-ahead bias.
- **Streamlit dashboard** - live signal, macro dashboard, SHAP feature importance, L/S PnL

## Results

| Metric | Value | Baseline |
|--------|-------|----------|
| Directional Accuracy | **65.4%** | 50% (random) |
| Long-Short Sharpe Ratio | **2.42** | 0.0 |
| R2 Score | ~0.009 | - |
| Test Period | 52 weeks | - |

Note: low R2 is expected and normal for financial return prediction.
Directional accuracy and Sharpe ratio are the relevant metrics, not R2.

## Why the Top Feature Makes Sense

**Inventory surprise** (actual EIA crude stocks vs 5-year seasonal average) was the
most predictive feature - and this maps directly to how commodity desks think.
Traders don't care about absolute inventory levels; they care whether supply/demand
balance is better or worse than expected for that time of year. The Wednesday EIA
report moves oil markets more than almost any other regular release.

## Setup

```bash
pip install -r requirements.txt

# Set free API keys for live macro + inventory data (optional but recommended)
# FRED key: https://fred.stlouisfed.org/docs/api/api_key.html
# EIA key:  https://www.eia.gov/opendata/
set FRED_API_KEY=your_key_here
set EIA_API_KEY=your_key_here

# Run model (works without API keys using synthetic macro data)
python run.py

# Launch interactive dashboard
python -m streamlit run dashboard.py
```

## Tech Stack

`Python` `XGBoost` `SHAP` `pandas` `NumPy` `yfinance` `FRED API` `EIA API` `Streamlit` `Plotly`

## Project Structure

```
wti-crude-oil-forecasting/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- run.py                     # CLI runner
|-- dashboard.py               # Streamlit app (5 tabs)
|-- __init__.py
|-- config.py                  # API series IDs, feature docs, model params
|-- data_pipeline.py           # yfinance + FRED + EIA fetch, mock generator
|-- feature_engineering.py     # 23 features with economic rationale
|-- model.py                   # XGBoost + walk-forward backtest
`-- xgb_oil_model.pkl          # Saved trained model (665 KB)
```

## What I Learned

Walk-forward backtesting was the hardest part to get right.
The naive train/test split is misleading - it tests one model trained at one point
and pretends it works for all future weeks. Walk-forward replicates reality:
you retrain as new data arrives. Getting this right is the difference between
a trustworthy result (65.4% DA) and an optimistic one.

Also: low R2 is NOT a failure for financial forecasting. Returns are noisy by design.
What matters is directional accuracy and whether the strategy generates positive Sharpe.

## Disclaimer

Research model for learning purposes only. Not financial advice.
