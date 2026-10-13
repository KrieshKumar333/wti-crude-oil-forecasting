"""
data_pipeline.py — Fetches and merges all raw data for the oil forecasting model.

Data sources (all free):
  1. Yahoo Finance (yfinance) — WTI/Brent prices, DXY, VIX, S&P 500
  2. FRED API            — Treasury yields, CPI, GDP (optional, needs key)
  3. EIA API             — US crude inventories, production (optional, needs key)
  4. Mock generator      — Realistic synthetic data when live APIs unavailable

Architecture: fetch each source independently → resample to weekly → merge on date index.
Missing values forward-filled then back-filled (never imputed across long gaps).
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import (
    WTI_TICKER, BRENT_TICKER, DXY_TICKER, VIX_TICKER, SP500_TICKER,
    FRED_SERIES, EIA_SERIES, HISTORY_YEARS, MOCK_WTI_BASE_PRICE, MOCK_SEED,
)

logger = logging.getLogger(__name__)

_FRED_KEY_ENV = "FRED_API_KEY"    # set in environment: export FRED_API_KEY=your_key
_EIA_KEY_ENV  = "EIA_API_KEY"     # set in environment: export EIA_API_KEY=your_key


# ─────────────────────────────────────────────────────────────────────────────
class DataPipeline:
    """
    Orchestrates fetching and merging of all raw data.

    Usage:
        pipeline = DataPipeline(use_mock=False)
        df = pipeline.get_master_dataframe()
    """

    def __init__(self, use_mock: bool = False,
                 fred_key: str = "", eia_key: str = "") -> None:
        self.use_mock   = use_mock
        # Accept keys directly OR fall back to environment variable
        self.fred_key   = fred_key or os.environ.get(_FRED_KEY_ENV, "")
        self.eia_key    = eia_key  or os.environ.get(_EIA_KEY_ENV,  "")
        self._cache: Optional[pd.DataFrame] = None
        self.start_date = datetime.today() - timedelta(weeks=52 * HISTORY_YEARS)
        self.end_date   = datetime.today()

    # ── Public interface ──────────────────────────────────────────────────────

    def get_master_dataframe(self) -> pd.DataFrame:
        """Return the merged weekly dataframe of all features and target."""
        if self._cache is not None:
            return self._cache

        if self.use_mock:
            df = self._generate_mock_data()
        else:
            df = self._fetch_live_data()

        df = self._validate_and_clean(df)
        self._cache = df
        logger.info(f"Master dataframe: {len(df)} weekly rows | {len(df.columns)} raw columns | "
                    f"from {df.index[0].date()} to {df.index[-1].date()}")
        return df

    # ── Live data fetching ────────────────────────────────────────────────────

    def _fetch_live_data(self) -> pd.DataFrame:
        """Fetch all live data sources and merge into a single weekly DataFrame."""
        frames: Dict[str, pd.Series] = {}

        # ── Yahoo Finance prices ──────────────────────────────────────────────
        logger.info("Fetching Yahoo Finance price data…")
        for name, ticker in [
            ("wti",  WTI_TICKER),
            ("brent",BRENT_TICKER),
            ("dxy",  DXY_TICKER),
            ("vix",  VIX_TICKER),
            ("sp500",SP500_TICKER),
        ]:
            try:
                raw = yf.download(ticker, start=self.start_date, end=self.end_date)["Close"].squeeze()
                # Resample to weekly (Friday close)
                frames[name] = raw.resample("W-FRI").last().rename(name)
                logger.info(f"  {ticker}: {len(frames[name])} weekly bars")
            except Exception as e:
                logger.warning(f"  Failed {ticker}: {e} — will use mock for this series")
                frames[name] = None  # type: ignore

        # Check if we got the critical WTI series
        if frames.get("wti") is None:
            logger.warning("WTI data unavailable — switching to full mock mode")
            return self._generate_mock_data()

        # ── FRED macro data (optional) ────────────────────────────────────────
        fred_key = self.fred_key
        if fred_key:
            logger.info("Fetching FRED macro data…")
            for col_name, series_id in FRED_SERIES.items():
                try:
                    s = self._fetch_fred_series(fred_key, series_id)
                    frames[col_name] = s.resample("W-FRI").last().rename(col_name)
                    logger.info(f"  FRED {series_id}: {len(frames[col_name])} obs")
                except Exception as e:
                    logger.warning(f"  FRED {series_id} failed: {e}")
        else:
            logger.info("No FRED_API_KEY found — generating synthetic yield/CPI data")
            frames.update(self._mock_macro_series(frames["wti"].index))

        # ── EIA inventory data (optional) ────────────────────────────────────
        eia_key  = self.eia_key
        if eia_key:
            logger.info("Fetching EIA inventory data…")
            for col_name, series_id in EIA_SERIES.items():
                try:
                    s = self._fetch_eia_series(eia_key, series_id)
                    frames[col_name] = s.resample("W-FRI").last().rename(col_name)
                    logger.info(f"  EIA {series_id}: {len(frames[col_name])} obs")
                except Exception as e:
                    logger.warning(f"  EIA {series_id} failed: {e}")
        else:
            logger.info("No EIA_API_KEY found — generating synthetic inventory data")
            frames.update(self._mock_eia_series(frames["wti"].index))

        # ── Merge on common weekly index ──────────────────────────────────────
        valid_frames = {k: v for k, v in frames.items() if v is not None}
        df = pd.DataFrame(valid_frames)
        df = df.resample("W-FRI").last()  # ensure consistent weekly index
        logger.info(f"Merged dataframe: {df.shape}")
        return df

    # ── API fetch helpers ─────────────────────────────────────────────────────

    def _fetch_fred_series(self, api_key: str, series_id: str) -> pd.Series:
        """Fetch one FRED series via REST API."""
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={api_key}&file_type=json"
            f"&observation_start={self.start_date.strftime('%Y-%m-%d')}"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        obs = r.json()["observations"]
        dates  = [o["date"] for o in obs if o["value"] != "."]
        values = [float(o["value"]) for o in obs if o["value"] != "."]
        return pd.Series(values, index=pd.to_datetime(dates), name=series_id)

    def _fetch_eia_series(self, api_key: str, series_id: str) -> pd.Series:
        """Fetch one EIA weekly series via REST API (v2)."""
        url = (
            f"https://api.eia.gov/v2/seriesid/{series_id}"
            f"?api_key={api_key}&out=json"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()["response"]["data"]
        df = pd.DataFrame(data)
        df["period"] = pd.to_datetime(df["period"])
        df["value"]  = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("period")
        return pd.Series(df["value"].values, index=df["period"], name=series_id)

    # ── Mock data generators ──────────────────────────────────────────────────

    def _generate_mock_data(self) -> pd.DataFrame:
        """
        Generates realistic synthetic weekly oil market data.
        Uses a regime-switching GBM for WTI, correlated macro processes for
        DXY/VIX/yields, and mean-reverting process for inventories.
        """
        np.random.seed(MOCK_SEED)
        idx = pd.date_range(end=datetime.today(), periods=HISTORY_YEARS * 52, freq="W-FRI")
        n_weeks = len(idx)   # use actual length after date_range construction

        # ── WTI price — regime-switching GBM ─────────────────────────────────
        regimes = np.random.choice(["bull", "bear", "sideways"],
                                   size=n_weeks,
                                   p=[0.40, 0.25, 0.35])
        wti_rets = np.where(
            regimes == "bull",   np.random.normal(0.004,  0.025, n_weeks),
            np.where(
            regimes == "bear",   np.random.normal(-0.003, 0.030, n_weeks),
                                 np.random.normal(0.000,  0.018, n_weeks)
            )
        )
        # Anchor last simulated price to MOCK_WTI_BASE_PRICE by subtracting terminal cumsum
        wti = MOCK_WTI_BASE_PRICE * np.exp(np.cumsum(wti_rets) - np.cumsum(wti_rets)[-1])
        wti = np.clip(wti, 30, 150)

        # ── Brent — WTI + spread ─────────────────────────────────────────────
        spread = np.random.normal(2.5, 1.2, n_weeks)
        brent  = wti + spread

        # ── DXY — negative correlation with oil ──────────────────────────────
        dxy_rets = -0.3 * wti_rets + np.random.normal(0, 0.005, n_weeks)
        dxy = 103 * np.exp(np.cumsum(dxy_rets) - np.cumsum(dxy_rets)[-1])
        dxy = np.clip(dxy, 88, 120)

        # ── VIX — spikes during bear regimes ─────────────────────────────────
        vix = 18 + np.where(regimes == "bear", 12, 0) + np.random.normal(0, 3, n_weeks)
        vix = np.clip(np.abs(vix), 10, 65)

        # ── S&P 500 ───────────────────────────────────────────────────────────
        sp500_rets = 0.001 - 0.5 * wti_rets * 0 + np.random.normal(0.001, 0.022, n_weeks)
        sp500 = 4500 * np.exp(np.cumsum(sp500_rets) - np.cumsum(sp500_rets)[-1] + np.log(4500))

        # ── 10Y / 2Y Treasury yields ─────────────────────────────────────────
        us_10y = 4.2 + np.cumsum(np.random.normal(0, 0.015, n_weeks))
        us_10y = np.clip(us_10y, 1.5, 8.0)
        us_2y  = us_10y - np.abs(np.random.normal(0.3, 0.5, n_weeks))

        # ── Crude inventory (mean-reverting around 430 million bbl) ──────────
        inv  = np.zeros(n_weeks)
        inv[0] = 430_000  # thousand bbls
        for i in range(1, n_weeks):
            inv[i] = 0.95 * inv[i-1] + 0.05 * 430_000 + np.random.normal(0, 4_000)
        inv = np.clip(inv, 380_000, 500_000)

        # ── US crude production (gradual uptrend + noise) ────────────────────
        prod = 12_500 + np.linspace(0, 800, n_weeks) + np.random.normal(0, 150, n_weeks)
        prod = np.clip(prod, 10_000, 14_500)

        df = pd.DataFrame({
            "wti":            wti,
            "brent":          brent,
            "dxy":            dxy,
            "vix":            vix,
            "sp500":          sp500,
            "us_10y_yield":   us_10y,
            "us_2y_yield":    us_2y,
            "us_crude_inventory":  inv,
            "us_crude_production": prod,
        }, index=idx)

        logger.info(f"Mock data generated: {len(df)} weekly bars")
        return df

    def _mock_macro_series(self, index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """Generate synthetic FRED macro series when API key unavailable."""
        np.random.seed(MOCK_SEED + 10)
        n = len(index)
        return {
            "us_10y_yield": pd.Series(
                np.clip(4.2 + np.cumsum(np.random.normal(0, 0.015, n)), 1.5, 8.0), index=index),
            "us_2y_yield": pd.Series(
                np.clip(3.8 + np.cumsum(np.random.normal(0, 0.018, n)), 0.5, 8.5), index=index),
        }

    def _mock_eia_series(self, index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """Generate synthetic EIA inventory/production data when API key unavailable."""
        np.random.seed(MOCK_SEED + 20)
        n = len(index)
        inv  = 430_000 + np.cumsum(np.random.normal(0, 3_500, n))
        inv  = np.clip(inv, 380_000, 500_000)
        prod = 12_500 + np.linspace(0, 800, n) + np.random.normal(0, 150, n)
        return {
            "us_crude_inventory":  pd.Series(inv,  index=index, name="us_crude_inventory"),
            "us_crude_production": pd.Series(prod, index=index, name="us_crude_production"),
        }

    # ── Validation & cleaning ─────────────────────────────────────────────────

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Forward-fill then back-fill (max 3 periods) — for holidays/API gaps
        2. Drop rows where WTI price is missing — can't work without the target
        3. Clip extreme outliers (>5 std devs) for numeric stability
        """
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Fill gaps (max 3 consecutive missing values)
        df = df.ffill(limit=3).bfill(limit=3)

        # Drop rows with missing WTI (our core variable)
        df = df.dropna(subset=["wti"])

        # Clip outliers — 5 sigma
        for col in df.select_dtypes(include=np.number).columns:
            mu, sigma = df[col].mean(), df[col].std()
            if sigma > 0:
                df[col] = df[col].clip(lower=mu - 5*sigma, upper=mu + 5*sigma)

        logger.info(f"Data clean: {len(df)} rows, {df.isna().sum().sum()} remaining NaNs")
        return df
