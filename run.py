"""
run.py — Batch runner for Crude Oil Forecasting Model.
Usage:
    python run.py           # live data (needs API keys in env)
    python run.py --mock    # synthetic data, no internet required
"""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run")


def main(use_mock: bool = False) -> None:
    from src.data_pipeline import DataPipeline
    from src.feature_engineering import FeatureEngineer
    from src.model import OilForecastModel

    logger.info("=" * 60)
    logger.info("Crude Oil Price Forecasting Model")
    logger.info("=" * 60)

    # 1. Data
    pipeline = DataPipeline(use_mock=use_mock)
    df_raw   = pipeline.get_master_dataframe()
    logger.info(f"Raw data: {df_raw.shape} | {df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    # 2. Features
    eng = FeatureEngineer(df_raw)
    X, y, feat_names = eng.build()
    logger.info(f"Feature matrix: {X.shape} | Target mean: {y.mean():.5f}")

    # 3. Model + backtest
    wti_prices = df_raw["wti"].reindex(X.index)
    mdl = OilForecastModel(X, y, feat_names, wti_prices)
    mdl.fit()
    bt  = mdl.run_backtest()

    logger.info("=" * 60)
    logger.info(f"Directional Accuracy : {bt.directional_acc:.1f}%  (baseline: 50%)")
    logger.info(f"Long/Short Sharpe    : {bt.sharpe_long_short:.2f}")
    logger.info(f"MAE (log-return)     : {bt.mae:.5f}")
    logger.info(f"RMSE (log-return)    : {bt.rmse:.5f}")
    logger.info(f"R² Score             : {bt.r2:.4f}")
    logger.info("=" * 60)

    # 4. Latest prediction
    pred = mdl.get_latest_prediction()
    logger.info(f"CURRENT SIGNAL  : {pred['direction']}")
    logger.info(f"Current WTI     : ${pred['current_wti_price']:.1f}/bbl")
    logger.info(f"4W Target Price : ${pred['predicted_price']:.1f}/bbl  ({pred['predicted_return_pct']:+.1f}%)")
    logger.info(f"Confidence      : {pred['confidence']}")

    # 5. Top features
    logger.info("\nTop 10 Features by Importance:")
    for feat, imp in bt.feature_importance.head(10).items():
        logger.info(f"  {feat:<30} {imp:.5f}")

    # 6. Save model
    import os; os.makedirs("outputs", exist_ok=True)
    mdl.save()
    logger.info("\nModel saved → outputs/xgb_oil_model.pkl")
    logger.info("Run 'streamlit run dashboard.py' to launch interactive dashboard.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use synthetic data")
    args = parser.parse_args()
    main(use_mock=args.mock)
