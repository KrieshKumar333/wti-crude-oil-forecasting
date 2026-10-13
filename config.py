"""
config.py — Configuration for Crude Oil Price Forecasting Model
Covers data sources, feature engineering, model hyperparameters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────

# Yahoo Finance tickers
WTI_TICKER   = "CL=F"     # WTI Crude Oil front-month futures
BRENT_TICKER = "BZ=F"     # Brent Crude Oil front-month futures
DXY_TICKER   = "DX-Y.NYB" # US Dollar Index
VIX_TICKER   = "^VIX"     # CBOE Volatility Index
SP500_TICKER = "^GSPC"    # S&P 500 (equity risk proxy)

# FRED series (Federal Reserve Economic Data) — requires FRED API key
# Free key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_SERIES = {
    "us_10y_yield":   "DGS10",       # 10-Year US Treasury Constant Maturity
    "us_2y_yield":    "DGS2",        # 2-Year US Treasury
    "cpi_us":         "CPIAUCSL",    # US Consumer Price Index
    "ism_mfg":        "MANEMP",      # US Manufacturing Employment (demand proxy)
    "us_gdp_growth":  "A191RL1Q225SBEA",  # US Real GDP growth
}

# EIA API (US Energy Information Administration)
# Free key at: https://www.eia.gov/opendata/
EIA_SERIES = {
    "us_crude_inventory": "PET.WCRSTUS1.W",   # US weekly crude stocks (thousand bbl)
    "us_crude_production":"PET.WCRFPUS2.W",   # US weekly crude production (thousand bbl/d)
    "us_crude_imports":   "PET.WCRIMUS2.W",   # US weekly crude imports (thousand bbl/d)
    "us_refinery_runs":   "PET.WCRRIUS2.W",   # US weekly refinery utilisation
}

# How many years of history to fetch
HISTORY_YEARS = 6

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE DEFINITIONS
#  Each feature is documented with its economic rationale for GIR interviews
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DOCS: Dict[str, str] = {
    # ── Price-based features ─────────────────────────────────────────────────
    "wti_ret_1w":    "1-week WTI return — short-term momentum signal",
    "wti_ret_4w":    "4-week WTI return — medium-term trend",
    "wti_ret_13w":   "13-week WTI return — quarterly trend (seasonality proxy)",
    "wti_vol_4w":    "4-week realised volatility — regime indicator",
    "wti_vol_13w":   "13-week realised volatility — market uncertainty",
    "brent_wti_spread": "Brent-WTI spread — geopolitical risk / logistics premium",
    "wti_52w_zscore":"Price z-score vs 52-week range — mean-reversion signal",

    # ── Inventory features ───────────────────────────────────────────────────
    "inventory_change_4w": "4-week US crude inventory change — supply-demand balance",
    "inventory_surprise":  "Inventory vs 5-year seasonal average — bullish/bearish signal",
    "inventory_mom":       "Inventory month-on-month change",
    "days_of_supply":      "Inventory / daily demand — key OPEC/IEA metric",

    # ── Production & macro features ─────────────────────────────────────────
    "us_production_chg":   "US crude production change — shale supply signal",
    "dxy_ret_4w":          "Dollar index 4-week return — inverse relationship with oil",
    "dxy_vol_4w":          "Dollar volatility — currency risk signal",
    "vix_level":           "VIX level — risk-off / risk-on environment",
    "vix_4w_change":       "VIX 4-week change — sentiment shift",
    "us_10y_yield":        "10-Year Treasury yield — inflation expectations, real rates",
    "yield_curve_spread":  "10Y minus 2Y spread — recession indicator",
    "sp500_ret_4w":        "S&P 500 4-week return — global risk appetite proxy",

    # ── Seasonal & calendar features ────────────────────────────────────────
    "month_sin":           "sin(month) — seasonal cyclicality encoding",
    "month_cos":           "cos(month) — seasonal cyclicality encoding",
    "is_q3":               "Q3 indicator (Jul-Sep) — driving season / peak demand",
    "is_q1":               "Q1 indicator (Jan-Mar) — heating season",
}

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PREDICTION_HORIZON = 4    # weeks ahead to forecast
TARGET_COL         = "wti_4w_fwd_return"   # 4-week forward log-return of WTI
TRAIN_TEST_SPLIT   = 0.80  # 80% train, 20% test

# XGBoost hyperparameters (tuned for commodity forecasting)
XGBOOST_PARAMS: Dict = {
    "n_estimators":       500,
    "max_depth":          4,
    "learning_rate":      0.03,
    "subsample":          0.8,
    "colsample_bytree":   0.7,
    "min_child_weight":   5,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "objective":          "reg:squarederror",
    "random_state":       42,
    "n_jobs":             -1,
}

# Rolling backtest window
BACKTEST_WINDOW_WEEKS = 104   # 2-year backtest period

# ─────────────────────────────────────────────────────────────────────────────
#  MOCK DATA PARAMETERS
#  Used when live APIs are unavailable (e.g., missing API keys in testing)
# ─────────────────────────────────────────────────────────────────────────────
MOCK_WTI_BASE_PRICE = 82.5    # USD/bbl approximate recent price
MOCK_SEED           = 42

# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT PATHS
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_OUTPUT   = "outputs/Oil_Forecasting_Report.xlsx"
PDF_OUTPUT     = "outputs/Oil_Forecasting_Note.pdf"
MODEL_PICKLE   = "outputs/xgb_oil_model.pkl"
