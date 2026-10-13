"""
feature_engineering.py — Constructs all model features from raw price/macro data.

Every feature has an explicit economic rationale (see config.FEATURE_DOCS).
All features are returns, z-scores, or level indicators — no raw prices leaked
into model inputs, preventing look-ahead bias.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PREDICTION_HORIZON, TARGET_COL

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class FeatureEngineer:
    """
    Transforms the raw merged DataFrame into model-ready features + target.

    Call .build() to get (X, y, feature_names).
    """

    def __init__(self, df_raw: pd.DataFrame) -> None:
        self.raw  = df_raw.copy()
        self._df: Optional[pd.DataFrame] = None

    def build(self) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Returns (X, y, feature_names):
          X — feature matrix (no lookahead)
          y — PREDICTION_HORIZON-week forward log-return of WTI
          feature_names — ordered list of feature column names
        """
        df = self._engineer_features()
        df = self._add_target(df)
        df = df.dropna()

        feature_cols = [c for c in df.columns if c != TARGET_COL and c != "wti"]
        X = df[feature_cols]
        y = df[TARGET_COL]

        logger.info(f"Feature matrix: {X.shape} | Features: {len(feature_cols)}")
        return X, y, feature_cols

    # ── Feature construction ──────────────────────────────────────────────────

    def _engineer_features(self) -> pd.DataFrame:
        df = self.raw.copy()
        wti   = df["wti"]
        brent = df.get("brent",   wti + 2.5)
        dxy   = df.get("dxy",     pd.Series(103.0, index=df.index))
        vix   = df.get("vix",     pd.Series(18.0,  index=df.index))
        sp500 = df.get("sp500",   pd.Series(4500.0, index=df.index))
        y10   = df.get("us_10y_yield", pd.Series(4.2,  index=df.index))
        y2    = df.get("us_2y_yield",  pd.Series(3.8,  index=df.index))
        inv   = df.get("us_crude_inventory",  pd.Series(430_000.0, index=df.index))
        prod  = df.get("us_crude_production", pd.Series(12_500.0,  index=df.index))

        feats = pd.DataFrame(index=df.index)

        # ── WTI momentum ─────────────────────────────────────────────────────
        feats["wti_ret_1w"]  = np.log(wti / wti.shift(1))
        feats["wti_ret_4w"]  = np.log(wti / wti.shift(4))
        feats["wti_ret_13w"] = np.log(wti / wti.shift(13))

        # ── WTI realised volatility ────────────────────────────────────────
        log_ret = np.log(wti / wti.shift(1))
        feats["wti_vol_4w"]  = log_ret.rolling(4).std()
        feats["wti_vol_13w"] = log_ret.rolling(13).std()

        # ── Spread & z-score ──────────────────────────────────────────────
        feats["brent_wti_spread"] = brent - wti
        roll_mean = wti.rolling(52).mean()
        roll_std  = wti.rolling(52).std()
        feats["wti_52w_zscore"]   = (wti - roll_mean) / roll_std.replace(0, np.nan)

        # ── Inventory ─────────────────────────────────────────────────────
        feats["inventory_change_4w"] = inv.diff(4)
        # Inventory surprise vs 5-year seasonal average
        inv_df = inv.to_frame("inv")
        inv_df["week_of_year"] = inv_df.index.isocalendar().week.astype(int)
        seasonal_avg = inv_df.groupby("week_of_year")["inv"].transform(
            lambda x: x.rolling(min_periods=1, window=260).mean()
        )
        feats["inventory_surprise"] = inv - seasonal_avg
        feats["inventory_mom"]      = inv.pct_change(4)
        feats["days_of_supply"]     = inv / (prod * 7)  # inventory / weekly demand

        # ── Production ────────────────────────────────────────────────────
        feats["us_production_chg"] = prod.pct_change(4)

        # ── DXY (dollar) ──────────────────────────────────────────────────
        dxy_log = np.log(dxy / dxy.shift(1))
        feats["dxy_ret_4w"]  = np.log(dxy / dxy.shift(4))
        feats["dxy_vol_4w"]  = dxy_log.rolling(4).std()

        # ── Risk sentiment ────────────────────────────────────────────────
        feats["vix_level"]     = vix
        feats["vix_4w_change"] = vix.diff(4)

        # ── Rates ─────────────────────────────────────────────────────────
        feats["us_10y_yield"]     = y10
        feats["yield_curve_spread"] = y10 - y2

        # ── Equities (risk-on/off proxy) ──────────────────────────────────
        feats["sp500_ret_4w"] = np.log(sp500 / sp500.shift(4))

        # ── Seasonal encoding (cyclic) ────────────────────────────────────
        month = df.index.month
        feats["month_sin"] = np.sin(2 * np.pi * month / 12)
        feats["month_cos"] = np.cos(2 * np.pi * month / 12)
        feats["is_q3"]     = (month.isin([7, 8, 9])).astype(int)
        feats["is_q1"]     = (month.isin([1, 2, 3])).astype(int)

        # Keep WTI price for target construction (dropped from X later)
        feats["wti"] = wti

        self._df = feats
        return feats

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Target: PREDICTION_HORIZON-week forward log-return of WTI.
        Shifted back by horizon so that row t has the return observed
        from t to t+horizon — no look-ahead bias.
        """
        fwd_ret = np.log(df["wti"].shift(-PREDICTION_HORIZON) / df["wti"])
        df[TARGET_COL] = fwd_ret
        return df

    def get_feature_importance_names(self) -> List[str]:
        """Return feature names for SHAP plots."""
        if self._df is None:
            raise RuntimeError("Call build() first")
        return [c for c in self._df.columns if c not in (TARGET_COL, "wti")]
