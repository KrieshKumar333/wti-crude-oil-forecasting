"""
dashboard.py — WTI Crude Oil Forecasting Dashboard
Run: python -m streamlit run dashboard.py
"""
from __future__ import annotations
import logging, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.config import PREDICTION_HORIZON, FEATURE_DOCS
from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.model import OilForecastModel

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WTI Crude Oil Forecasting",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛢️ WTI Crude Oil")
    st.caption("Macro-Quantitative Forecasting Model")
    st.divider()

    use_mock = st.toggle("Use offline demo data", value=False,
                         help="Enable if no internet. Uses synthetic but statistically realistic data.")
    st.divider()

    st.subheader("🔑 API Keys")
    st.caption("Paste your free keys below. Keys are session-only — never stored.")

    fred_key = st.text_input(
        "FRED API Key",
        type="password",
        placeholder="e.g. abcd1234efgh5678...",
        help="Get free at: https://fred.stlouisfed.org/docs/api/api_key.html — provides Treasury yields, CPI, GDP",
    )
    eia_key = st.text_input(
        "EIA API Key",
        type="password",
        placeholder="e.g. ABCD1234EFGH5678...",
        help="Get free at: https://www.eia.gov/opendata/ — provides US crude inventory & production data",
    )

    if fred_key:
        st.success("✓ FRED key entered")
    else:
        st.warning("No FRED key — using synthetic macro data")

    if eia_key:
        st.success("✓ EIA key entered")
    else:
        st.warning("No EIA key — using synthetic inventory data")

    st.divider()
    st.subheader("ℹ️ How to get free keys")
    with st.expander("FRED (Treasury yields, CPI)"):
        st.markdown("""
1. Go to **fred.stlouisfed.org**
2. Click **My Account → API Keys**
3. Click **Request API Key**
4. Fill name + reason → key shown instantly
""")
    with st.expander("EIA (Oil inventory, production)"):
        st.markdown("""
1. Go to **eia.gov/opendata**
2. Click **Register**
3. Fill the form
4. Key emailed within 5 minutes
""")

# ── Load data & run model ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading market data…")
def run_pipeline(use_mock_: bool, fred_key_: str, eia_key_: str):
    pipeline = DataPipeline(use_mock=use_mock_, fred_key=fred_key_, eia_key=eia_key_)
    df_raw   = pipeline.get_master_dataframe()
    eng      = FeatureEngineer(df_raw)
    X, y, feat_names = eng.build()
    wti_prices = df_raw["wti"].reindex(X.index)
    mdl = OilForecastModel(X, y, feat_names, wti_prices)
    mdl.fit()
    bt  = mdl.run_backtest()
    pred = mdl.get_latest_prediction()
    return df_raw, X, mdl, bt, pred, wti_prices

with st.spinner("Loading oil market data and training model…"):
    try:
        df_raw, X, mdl, bt, pred, wti_prices = run_pipeline(use_mock, fred_key, eia_key)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.info("Try enabling 'Use offline demo data' in the sidebar.")
        st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛢️ WTI Crude Oil — Macro Forecasting Model")
data_src = "Offline Demo" if use_mock else f"Live · Yahoo Finance{' + FRED' if fred_key else ''}{' + EIA' if eia_key else ''}"
st.caption(f"XGBoost | Walk-forward backtest | 23 features | {PREDICTION_HORIZON}-week forecast | {data_src}")
st.divider()

# ── KPI strip ─────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
direction = pred["direction"]
dir_color = "green" if "↑" in direction else "red"
k1.metric("Current WTI",       f"${pred['current_wti_price']:.1f}/bbl")
k2.metric(f"{PREDICTION_HORIZON}W Target",
          f"${pred['predicted_price']:.1f}/bbl",
          f"{pred['predicted_return_pct']:+.1f}%")
k3.metric("Signal",            f":{dir_color}[{direction}]")
k4.metric("Directional Acc.",  f"{bt.directional_acc:.1f}%", "vs 50% random")
k5.metric("L/S Sharpe",        f"{bt.sharpe_long_short:.2f}")
k6.metric("Confidence",        pred["confidence"])
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price & Forecast",
    "🔁 Backtest Performance",
    "🔬 Feature Importance",
    "📊 Macro Dashboard",
    "📋 Model Summary",
])

# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        wti_plot = df_raw["wti"].iloc[-156:]
        fig = go.Figure()
        fig.add_scatter(x=wti_plot.index, y=wti_plot.values,
                        line=dict(color="#1f77b4", width=1.8),
                        fill="tozeroy", fillcolor="rgba(31,119,180,0.08)",
                        name="WTI (USD/bbl)")
        last_date = wti_plot.index[-1]
        pred_date = last_date + pd.Timedelta(weeks=PREDICTION_HORIZON)
        fig.add_shape(type="line", x0=last_date, x1=last_date, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="orange", dash="dash", width=1.5))
        forecast_color = "#2ca02c" if pred["predicted_return_pct"] > 0 else "#d62728"
        fig.add_scatter(x=[last_date, pred_date],
                        y=[pred["current_wti_price"], pred["predicted_price"]],
                        line=dict(color=forecast_color, dash="dot", width=2.5),
                        mode="lines+markers",
                        name=f"{PREDICTION_HORIZON}W Forecast")
        fig.update_layout(title="WTI Crude Oil — 3-Year Price History + Forecast",
                          yaxis_title="USD per Barrel",
                          template="plotly_white", height=380,
                          hovermode="x unified",
                          margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Current Signal")
        if "↑" in direction:
            st.success(f"### {direction}")
        else:
            st.error(f"### {direction}")
        st.markdown(f"**{PREDICTION_HORIZON}W Return:** `{pred['predicted_return_pct']:+.2f}%`")
        st.markdown(f"**Target Price:** `${pred['predicted_price']:.1f}/bbl`")
        st.markdown(f"**Confidence:** `{pred['confidence']}`")
        st.markdown(f"**As of:** {pred['prediction_date']}")
        st.divider()
        st.subheader("Model Stats")
        st.markdown(f"Dir. Accuracy: **{bt.directional_acc:.1f}%**")
        st.markdown(f"L/S Sharpe: **{bt.sharpe_long_short:.2f}**")
        st.markdown(f"R² Score: **{bt.r2:.4f}**")
        st.markdown(f"Test weeks: **{len(bt.y_true)}**")

    # Scenario bridge
    st.subheader("Signal Interpretation")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(f"**Model Input**\n\nCurrent WTI: **${pred['current_wti_price']:.1f}/bbl**")
    with col_b:
        if pred["predicted_return_pct"] > 0:
            st.success(f"**4-Week Forecast**\n\n**${pred['predicted_price']:.1f}/bbl** ({pred['predicted_return_pct']:+.1f}%)")
        else:
            st.error(f"**4-Week Forecast**\n\n**${pred['predicted_price']:.1f}/bbl** ({pred['predicted_return_pct']:+.1f}%)")
    with col_c:
        st.info(f"**Historical Accuracy**\n\n**{bt.directional_acc:.1f}%** correct direction\n(baseline: 50%)")

# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Walk-Forward Backtest — Expanding Window")
    st.caption("Each test week: trained on ALL prior data, predicts 4 weeks ahead. Zero look-ahead bias.")

    c1, c2 = st.columns(2)
    with c1:
        fig_bt = go.Figure()
        fig_bt.add_scatter(x=bt.dates, y=bt.y_true*100,
                           line=dict(color="#1f77b4", width=1.2), name="Actual Return %")
        fig_bt.add_scatter(x=bt.dates, y=bt.y_pred*100,
                           line=dict(color="#ff7f0e", width=1.5, dash="dot"), name="Predicted Return %")
        fig_bt.add_hline(y=0, line_dash="dash", line_color="grey")
        fig_bt.update_layout(title="Predicted vs Actual 4-Week Returns",
                             yaxis_title="Return (%)", template="plotly_white",
                             height=350, margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig_bt, use_container_width=True)

    with c2:
        correct = int((np.sign(bt.y_pred) == np.sign(bt.y_true)).sum())
        wrong   = len(bt.y_true) - correct
        fig_pie = go.Figure(go.Pie(
            labels=["Correct Direction", "Wrong Direction"],
            values=[correct, wrong], hole=0.5,
            marker_colors=["#2ca02c", "#d62728"],
            textinfo="label+percent",
        ))
        fig_pie.update_layout(title=f"Directional Accuracy: {bt.directional_acc:.1f}%",
                              height=350, showlegend=False,
                              margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Cumulative L/S PnL
    ls_rets = np.where(bt.y_pred > 0, bt.y_true, -bt.y_true)
    cum_pnl = (np.cumprod(1 + ls_rets) - 1) * 100
    fig_pnl = go.Figure()
    fig_pnl.add_scatter(x=bt.dates, y=cum_pnl,
                        fill="tozeroy", fillcolor="rgba(44,160,44,0.15)",
                        line=dict(color="#2ca02c", width=2),
                        name="Cumulative L/S Return %")
    fig_pnl.add_hline(y=0, line_dash="dash", line_color="grey")
    fig_pnl.update_layout(title="Cumulative Long/Short Strategy Return (%)",
                          yaxis_title="Cumulative Return (%)",
                          template="plotly_white", height=300,
                          margin=dict(l=0,r=0,t=50,b=0))
    st.plotly_chart(fig_pnl, use_container_width=True)

    # Stats table
    stats = pd.DataFrame({
        "Metric": ["Directional Accuracy", "MAE (log-return)", "RMSE (log-return)",
                   "R² Score", "L/S Sharpe (annualised)", "Test Weeks", "Baseline (random)"],
        "Value":  [f"{bt.directional_acc:.1f}%", f"{bt.mae:.5f}", f"{bt.rmse:.5f}",
                   f"{bt.r2:.4f}", f"{bt.sharpe_long_short:.2f}", str(len(bt.y_true)), "50.0%"],
    }).set_index("Metric")
    st.dataframe(stats, use_container_width=True)

# ══ TAB 3 ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Feature Importance (XGBoost Gain)")
    top15 = bt.feature_importance.head(15)
    fig_fi = px.bar(x=top15.values, y=top15.index, orientation="h",
                    color=top15.values,
                    color_continuous_scale=[[0,"#e8f4fd"],[1,"#1f77b4"]],
                    title="Top 15 Features by Importance",
                    labels={"x":"Importance Score","y":""})
    fig_fi.update_layout(template="plotly_white", height=450,
                         showlegend=False, coloraxis_showscale=False,
                         yaxis=dict(autorange="reversed"),
                         margin=dict(l=0,r=0,t=50,b=0))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("Feature Descriptions")
    from src.config import FEATURE_DOCS
    for feat, doc in list(FEATURE_DOCS.items())[:12]:
        if feat in bt.feature_importance.index:
            rank = list(bt.feature_importance.index).index(feat) + 1
            st.markdown(f"**#{rank} `{feat}`** — {doc}")

    st.subheader("Feature Correlation Matrix (Top 12)")
    top12 = bt.feature_importance.head(12).index.tolist()
    avail = [f for f in top12 if f in X.columns]
    if avail:
        corr = X[avail].corr()
        fig_corr = px.imshow(corr, color_continuous_scale=[[0,"#d62728"],[0.5,"white"],[1,"#1f77b4"]],
                             zmin=-1, zmax=1, title="Feature Correlation Matrix")
        fig_corr.update_layout(height=450, margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig_corr, use_container_width=True)

# ══ TAB 4 ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Macro Environment Dashboard")
    macro_plots = [
        ("WTI vs DXY (Inverse Correlation)", "wti", "dxy"),
        ("WTI vs VIX (Risk-Off)", "wti", "vix"),
        ("WTI vs 10Y Treasury Yield", "wti", "us_10y_yield"),
    ]
    cols = st.columns(3)
    for i, (title, primary, secondary) in enumerate(macro_plots):
        with cols[i]:
            if secondary in df_raw.columns:
                fig = go.Figure()
                plot_data = df_raw[[primary, secondary]].iloc[-104:].dropna()
                fig.add_scatter(x=plot_data.index, y=plot_data[primary],
                                name="WTI", line=dict(color="#1f77b4", width=1.5), yaxis="y")
                fig.add_scatter(x=plot_data.index, y=plot_data[secondary],
                                name=secondary.replace("_"," ").title(),
                                line=dict(color="#ff7f0e", width=1.5, dash="dot"), yaxis="y2")
                fig.update_layout(title=title, template="plotly_white", height=270,
                                  legend=dict(x=0,y=1,font=dict(size=9)),
                                  yaxis=dict(title="WTI (USD/bbl)"),
                                  yaxis2=dict(title=secondary, overlaying="y", side="right"),
                                  margin=dict(l=0,r=50,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"{secondary} not available (enter API key for live data)")

    if "us_crude_inventory" in df_raw.columns:
        st.subheader("US Crude Inventory (thousand barrels)")
        inv = df_raw["us_crude_inventory"].iloc[-104:].dropna()
        fig_inv = go.Figure()
        fig_inv.add_scatter(x=inv.index, y=inv.values, fill="tozeroy",
                            fillcolor="rgba(255,127,14,0.12)",
                            line=dict(color="#ff7f0e", width=1.8))
        fig_inv.add_hline(y=inv.mean(), line_dash="dash", line_color="#1f77b4",
                          annotation_text=f"2Y Avg: {inv.mean():,.0f}")
        fig_inv.update_layout(yaxis_title="Thousand Barrels",
                              template="plotly_white", height=280,
                              showlegend=False, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig_inv, use_container_width=True)

# ══ TAB 5 ════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Model Summary — WTI Crude Oil Forecasting")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
#### Architecture
- **Model**: XGBoost (gradient boosted trees)
- **Target**: 4-week forward WTI log-return
- **Features**: 23 across 6 categories
- **Backtest**: Walk-forward expanding window (gold standard for time series)

#### Feature Categories
| Category | Features | Economic Rationale |
|----------|----------|-------------------|
| Momentum | 1W, 4W, 13W returns; 4W/13W vol | Price trend & regime |
| Inventory | 4W change, surprise vs seasonal, days-of-supply | Supply-demand balance |
| Dollar (DXY) | 4W return, 4W volatility | Inverse USD-oil link |
| Risk | VIX level, 4W change; S&P 500 4W return | Risk-on/off regime |
| Rates | 10Y yield, 10Y-2Y spread | Real rates, recession signal |
| Seasonal | sin/cos(month), Q1/Q3 dummies | Driving/heating season |

#### Why Walk-Forward?
A simple train/test split trains ONCE and predicts across all test dates — 
unrealistic for live trading. Walk-forward trains on all data up to week t, 
predicts t→t+4, then moves forward. No look-ahead bias at any step.
""")
    with col_r:
        st.markdown("#### Results")
        res_df = pd.DataFrame({
            "Metric": ["Directional Accuracy", "L/S Sharpe (ann.)",
                       "MAE (log-return)", "R² Score", "Test Period"],
            "Value":  [f"{bt.directional_acc:.1f}%", f"{bt.sharpe_long_short:.2f}",
                       f"{bt.mae:.5f}", f"{bt.r2:.4f}", f"{len(bt.y_true)} weeks"],
            "Benchmark": ["50.0% (random)", "0.0 (no signal)",
                          "—", "0.0", "—"],
        }).set_index("Metric")
        st.dataframe(res_df, use_container_width=True)

        st.markdown("#### Top 5 Features")
        for f, imp in bt.feature_importance.head(5).items():
            doc = FEATURE_DOCS.get(f, "")
            st.markdown(f"- **`{f}`** ({imp:.4f}) — {doc[:55]}...")

    st.caption("Research model for educational purposes only. Not financial advice.")
