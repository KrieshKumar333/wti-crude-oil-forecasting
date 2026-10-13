"""
model.py — XGBoost oil price forecasting model with walk-forward backtest.

Walk-forward (expanding window) backtest: for each week in the test set,
train on ALL data up to that week and predict PREDICTION_HORIZON weeks ahead.
This is the only backtest methodology that avoids look-ahead bias for
time-series data — a critical point for GIR interviews.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import (
    XGBOOST_PARAMS, TRAIN_TEST_SPLIT, PREDICTION_HORIZON,
    TARGET_COL, MODEL_PICKLE,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BacktestResults:
    dates:              pd.DatetimeIndex
    y_true:             np.ndarray
    y_pred:             np.ndarray
    directional_acc:    float           # % of weeks correct direction predicted
    mae:                float           # Mean Absolute Error (log-return)
    rmse:               float
    r2:                 float
    sharpe_long_short:  float           # Sharpe of a simple long/short strategy
    feature_importance: pd.Series       # Feature importances from final model
    wti_prices_actual:  pd.Series       # WTI price series for display
    n_train_final:      int             # Size of final training set


# ─────────────────────────────────────────────────────────────────────────────
class OilForecastModel:
    """
    XGBoost-based 4-week WTI crude oil return forecasting model.

    Walk-forward backtest on the last 20% of data (expanding window train set).
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        wti_prices: pd.Series,
    ) -> None:
        self.X             = X
        self.y             = y
        self.feature_names = feature_names
        self.wti_prices    = wti_prices
        self._model:   Optional[XGBRegressor] = None
        self._scaler:  Optional[StandardScaler] = None
        self._results: Optional[BacktestResults] = None

    # ── Training & prediction ─────────────────────────────────────────────────

    def fit(self) -> "OilForecastModel":
        """Fit the model on the full training set (first 80%)."""
        split = int(len(self.X) * TRAIN_TEST_SPLIT)
        X_train, y_train = self.X.iloc[:split], self.y.iloc[:split]

        self._scaler = StandardScaler()
        X_train_sc = self._scaler.fit_transform(X_train)

        self._model = XGBRegressor(**XGBOOST_PARAMS)
        self._model.fit(
            X_train_sc, y_train,
            eval_set=[(X_train_sc, y_train)],
            verbose=False,
        )
        logger.info(f"Model trained on {len(X_train)} samples | "
                    f"Features: {len(self.feature_names)}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict forward returns for given features."""
        if self._model is None or self._scaler is None:
            raise RuntimeError("Call fit() first")
        X_sc = self._scaler.transform(X)
        return self._model.predict(X_sc)

    # ── Walk-forward backtest ─────────────────────────────────────────────────

    def run_backtest(self) -> BacktestResults:
        """
        Walk-forward (expanding window) backtest on the test set.
        For each test week t:
          - Train on rows 0..t-1
          - Predict return at row t
          - Compare to actual return
        """
        if self._results is not None:
            return self._results

        split  = int(len(self.X) * TRAIN_TEST_SPLIT)
        n_test = len(self.X) - split

        y_true_list, y_pred_list, date_list = [], [], []

        logger.info(f"Running walk-forward backtest: {n_test} test weeks…")

        for i in range(split, len(self.X)):
            X_tr = self.X.iloc[:i]
            y_tr = self.y.iloc[:i]
            X_te = self.X.iloc[[i]]

            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)

            mdl = XGBRegressor(**XGBOOST_PARAMS)
            mdl.fit(X_tr_sc, y_tr, verbose=False)
            pred = float(mdl.predict(X_te_sc)[0])

            y_true_list.append(float(self.y.iloc[i]))
            y_pred_list.append(pred)
            date_list.append(self.X.index[i])

            if (i - split) % 20 == 0:
                logger.info(f"  Backtest progress: {i - split}/{n_test} weeks")

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        # ── Metrics ───────────────────────────────────────────────────────────
        directional_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true))) * 100
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2   = r2_score(y_true, y_pred)

        # Long/Short strategy Sharpe: long when pred>0, short when pred<0
        ls_rets = np.where(y_pred > 0, y_true, -y_true)  # 1x position
        sharpe  = (np.mean(ls_rets) / np.std(ls_rets)) * np.sqrt(52) \
                  if np.std(ls_rets) > 0 else np.nan

        # Feature importance from final model
        if self._model is None:
            self.fit()
        fi = pd.Series(
            self._model.feature_importances_,  # type: ignore
            index=self.feature_names,
        ).sort_values(ascending=False)

        # WTI prices for test period
        test_wti = self.wti_prices.iloc[split:]

        logger.info(
            f"Backtest complete: DA={directional_acc:.1f}% | "
            f"MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | "
            f"L/S Sharpe={sharpe:.2f}"
        )

        self._results = BacktestResults(
            dates              = pd.DatetimeIndex(date_list),
            y_true             = y_true,
            y_pred             = y_pred,
            directional_acc    = directional_acc,
            mae                = mae,
            rmse               = rmse,
            r2                 = r2,
            sharpe_long_short  = sharpe,
            feature_importance = fi,
            wti_prices_actual  = test_wti,
            n_train_final      = split,
        )
        return self._results

    def get_latest_prediction(self) -> Dict:
        """Predict the next PREDICTION_HORIZON weeks return from the most recent data."""
        if self._model is None:
            self.fit()
        X_last = self.X.iloc[[-1]]
        pred_return = float(self.predict(X_last)[0])
        current_price = float(self.wti_prices.iloc[-1])
        pred_price = current_price * np.exp(pred_return)
        direction = "BULLISH ↑" if pred_return > 0 else "BEARISH ↓"

        return {
            "prediction_date":      self.X.index[-1].strftime("%d %b %Y"),
            "forecast_horizon":     f"{PREDICTION_HORIZON} weeks",
            "current_wti_price":    current_price,
            "predicted_return_pct": pred_return * 100,
            "predicted_price":      pred_price,
            "direction":            direction,
            "confidence":           "High" if abs(pred_return) > 0.03 else "Moderate" if abs(pred_return) > 0.015 else "Low",
        }

    def save(self, path: str = MODEL_PICKLE) -> None:
        """Persist model + scaler for dashboard reuse."""
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler,
                         "feature_names": self.feature_names}, f)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> Dict:
        with open(path, "rb") as f:
            return pickle.load(f)
