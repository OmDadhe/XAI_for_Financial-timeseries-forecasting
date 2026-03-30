"""
╔══════════════════════════════════════════════════════════════════╗
║   Developing Explainable AI for Financial Time Series Forecasting ║
║   Streamlit Dashboard · Om Dadhe · GITAM University 2025         ║
╚══════════════════════════════════════════════════════════════════╝

Run locally:
    streamlit run app.py

Folder structure expected (same as notebook output):
    project_root/
    ├── app.py
    ├── requirements.txt
    ├── data/
    │   ├── raw/AAPL_synthesized_raw.csv
    │   └── processed/AAPL_processed.csv
    ├── reports/
    │   ├── initial_summary.txt
    │   ├── kpi_summary.txt
    │   └── final_summary.txt
    └── visualizations/
        ├── cleaning/
        ├── eda/
        ├── models/
        ├── shap/
        └── dashboard/
"""

import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI · Financial Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
DATA_RAW     = os.path.join(BASE, "data", "raw",       "AAPL_synthesized_raw.csv")
DATA_PROC    = os.path.join(BASE, "data", "processed",  "AAPL_processed.csv")
RPT_INITIAL  = os.path.join(BASE, "reports", "initial_summary.txt")
RPT_KPI      = os.path.join(BASE, "reports", "kpi_summary.txt")
RPT_FINAL    = os.path.join(BASE, "reports", "final_summary.txt")
VIZ          = os.path.join(BASE, "visualizations")

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def load_img(path: str):
    if os.path.exists(path):
        return Image.open(path)
    return None

def show_img(path: str, caption: str = "", use_container_width: bool = True):
    img = load_img(path)
    if img:
        st.image(img, caption=caption, use_container_width=use_container_width)
    else:
        st.info(f"📂 Image not found: `{os.path.basename(path)}`  \n"
                f"Run the notebook first to generate visualizations.")

def read_txt(path: str) -> str:
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return f"Report not found: {path}\nRun the notebook to generate reports."

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])
    return pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }

    /* KPI cards */
    .kpi-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 12px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 600;
        color: #58a6ff;
    }
    .kpi-sub {
        font-size: 11px;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #f0f6fc;
        border-left: 4px solid #58a6ff;
        padding-left: 12px;
        margin: 28px 0 16px 0;
    }

    /* Data story box */
    .story-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #3fb950;
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 14px;
        color: #c9d1d9;
        margin-bottom: 16px;
        font-style: italic;
    }

    /* Main background */
    .main .block-container {
        background: #0d1117;
        padding-top: 2rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 8px;
        gap: 4px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8b949e;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #21262d !important;
        color: #f0f6fc !important;
    }

    /* Code blocks */
    code {
        background: #161b22;
        color: #79c0ff;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 XAI · FinTS")
    st.markdown("---")
    st.markdown("**Developing Explainable AI**  \nfor Financial Time Series Forecasting")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "📊  EDA & Data Story",
            "🧠  Model Results",
            "🔬  SHAP Explainability",
            "🖥️  Final Dashboard",
            "📄  Reports",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("Synthesized AAPL · GBM · 10 Years")
    st.markdown("**Models**")
    st.markdown("LSTM+Attn · GRU · Bi-LSTM · XGBoost")
    st.markdown("**XAI**")
    st.markdown("SHAP Gradient + TreeSHAP")
    st.markdown("---")
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-OmDadhe-181717?logo=github)](https://github.com/OmDadhe/xai-financial-forecasting)  \n"
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-contactom-0077B5?logo=linkedin)](https://linkedin.com/in/contactom)"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    st.markdown("""
    # 📈 Developing Explainable AI  
    ### for Financial Time Series Forecasting
    """)
    st.markdown(
        "An end-to-end XAI framework on 10 years of synthesized AAPL-like stock data — "
        "multi-model benchmarking, SHAP-driven explainability, and temporal attribution at the feature level."
    )

    st.markdown("---")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    kpi_txt = read_txt(RPT_KPI)

    # Parse KPI values from txt if available
    def extract_kpi(txt, key):
        for line in txt.splitlines():
            if key in line:
                parts = line.split(":")
                if len(parts) > 1:
                    return parts[-1].strip()
        return "—"

    best_model = extract_kpi(kpi_txt, "Best model")
    best_rmse  = extract_kpi(kpi_txt, "Best RMSE")
    best_r2    = extract_kpi(kpi_txt, "Best R²")
    best_mape  = extract_kpi(kpi_txt, "Best MAPE")
    n_features = extract_kpi(kpi_txt, "Features used")
    window     = extract_kpi(kpi_txt, "Window size")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, label, val, sub in [
        (c1, "Best Model",    best_model, "by RMSE"),
        (c2, "Best RMSE",     best_rmse,  "lower is better"),
        (c3, "Best R²",       best_r2,    "higher is better"),
        (c4, "Best MAPE",     best_mape,  "% error"),
        (c5, "Features",      "12",       "OHLCV + 7 indicators"),
        (c6, "Window",        "60 days",  "lookback sequence"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Architecture cards ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Pipeline Architecture</div>', unsafe_allow_html=True)

    steps = [
        ("🧪", "Data Synthesis",     "GBM · μ=0.25 · σ=0.30 · 2,520 days"),
        ("🧹", "Cleaning & EDA",      "NaN interp · OHLC validation · 9 EDA charts"),
        ("⚙️", "Feature Engineering", "12 features · RSI · MACD · ATR · OBV · Stoch"),
        ("🔢", "Sequences",           "60-day rolling windows · MinMaxScaler"),
        ("🧠", "Model Training",      "LSTM+Attn · GRU · Bi-LSTM · XGBoost"),
        ("🔬", "SHAP XAI",           "GradientExplainer · TreeSHAP · Heatmaps"),
        ("🖥️", "Dashboard",           "Plotly HTML · Streamlit · Auto-saved reports"),
    ]

    cols = st.columns(len(steps))
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div class="kpi-card">
            <div style="font-size:28px">{icon}</div>
            <div style="font-weight:700;color:#f0f6fc;margin:8px 0 4px">{title}</div>
            <div style="font-size:12px;color:#8b949e">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Feature table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Set — 12 Inputs</div>', unsafe_allow_html=True)

    feat_df = pd.DataFrame([
        ("Open",     "Price",     "Gap risk from prior close"),
        ("High",     "Price",     "Intraday resistance level"),
        ("Low",      "Price",     "Intraday support level"),
        ("Close",    "Price",     "Primary prediction target ⭐"),
        ("Volume",   "Volume",    "Conviction behind price moves"),
        ("RSI-14",   "Momentum",  "Overbought >70 / Oversold <30"),
        ("MACD",     "Trend",     "Momentum convergence/divergence"),
        ("BB Width", "Volatility","Bollinger Band squeeze — precedes breakouts"),
        ("EMA-20",   "Trend",     "Exponential smoothing, responsive to recent price"),
        ("ATR-14",   "Volatility","Average True Range — raw volatility magnitude"),
        ("OBV",      "Volume",    "On-Balance Volume — trend via cumulative volume"),
        ("Stoch-%K", "Momentum",  "Position within recent high-low range"),
    ], columns=["Feature", "Type", "Financial Interpretation"])

    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Dataset preview ───────────────────────────────────────────────────────
    df_proc = load_csv(DATA_PROC)
    if not df_proc.empty:
        st.markdown('<div class="section-header">Processed Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df_proc.tail(10), use_container_width=True, hide_index=True)
    else:
        st.info("Run the notebook to generate `data/processed/AAPL_processed.csv`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & DATA STORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA & Data Story":

    st.markdown("# 📊 EDA & Data Story")
    st.markdown("Every visualization tells a specific analytical story. Expand each section to read it.")
    st.markdown("---")

    df_proc = load_csv(DATA_PROC)

    # ── Live Close Price Chart ────────────────────────────────────────────────
    if not df_proc.empty and "Close" in df_proc.columns:
        st.markdown('<div class="section-header">📈 Synthesized AAPL Close Price</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_proc["Date"], y=df_proc["Close"],
                                  line=dict(color="#58a6ff", width=1.2),
                                  name="Close", fill="tozeroy",
                                  fillcolor="rgba(88,166,255,0.07)"))
        if "MA_20" in df_proc.columns:
            fig.add_trace(go.Scatter(x=df_proc["Date"], y=df_proc["MA_20"],
                                      line=dict(color="#f7931e", width=1, dash="dot"),
                                      name="MA-20"))
        if "MA_200" in df_proc.columns:
            fig.add_trace(go.Scatter(x=df_proc["Date"], y=df_proc["MA_200"],
                                      line=dict(color="#e74c3c", width=1.5),
                                      name="MA-200"))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(x=0.01, y=0.99),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", title="Price (USD)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── EDA Sections ─────────────────────────────────────────────────────────
    eda_sections = [
        (
            "🧹 Data Cleaning",
            "cleaning_overview.png",
            "cleaning",
            "How bad was the raw data, and what did cleaning fix?",
            "Four-panel view: NaN heatmap (red = missing), raw vs cleaned close overlay, "
            "volume distribution before/after interpolation, and daily OHLC range post-cleaning. "
            "Missing values are randomly scattered — confirming noise artifacts, not structural gaps — "
            "justifying linear interpolation."
        ),
        (
            "📦 Univariate Distributions",
            "univariate_distributions.png",
            "eda",
            "What does the distribution of each OHLCV variable reveal about the data-generating process?",
            "Six histograms with mean/median overlays. Close and Open show right-skewed multimodal "
            "distributions — characteristic of a trending GBM asset. Daily returns approximate "
            "a normal distribution with slight leptokurtosis (fat tails), matching real equity markets."
        ),
        (
            "📦 Boxplots & Skewness",
            "boxplot_skewness.png",
            "eda",
            "Which features are most asymmetric, and does that matter for our scaler choice?",
            "OHLC boxplots reveal High has the widest range. The skewness/kurtosis bar chart confirms "
            "Volume is the most non-normal feature — justifying MinMaxScaler (robust to monotonic transforms) "
            "over StandardScaler."
        ),
        (
            "📈 Time Series Analysis",
            "time_series_analysis.png",
            "eda",
            "What are the structural patterns in price, volume, returns, and cumulative performance?",
            "Four-panel time series. Weekly OHLC bars show GBM drift clearly — price trends upward with "
            "increasing variance. Volume spikes visibly coincide with high-return days. "
            "Cumulative return exceeds 800% over the 10-year period."
        ),
        (
            "📅 Seasonality Analysis",
            "seasonality_analysis.png",
            "eda",
            "Are there calendar patterns in returns that a model should be aware of?",
            "Annual close shows consistent upward trend. Monthly returns reveal slight Q4 positive bias "
            "(October–November), consistent with the real-market Santa Claus rally. Day-of-week "
            "analysis shows Friday slightly underperforming — known in market microstructure."
        ),
        (
            "📉 Rolling Stats & Volatility",
            "rolling_stats_volatility.png",
            "eda",
            "Where are the volatility regimes, and do MA crossovers signal trend changes?",
            "MA-20, MA-50, and MA-200 overlaid on price reveal classic crossover points. Bollinger Bands "
            "(±2σ) capture expansion during high-volatility regimes. The 30-day annualised volatility "
            "panel exposes GARCH-like clustering — quiet periods followed by explosive moves."
        ),
        (
            "🔁 Autocorrelation",
            "autocorrelation.png",
            "eda",
            "Are returns serially correlated? Does volatility cluster?",
            "Returns ACF shows no significant autocorrelation beyond lag 1 (weak-form efficiency). "
            "Squared returns ACF shows strong persistence across 40 lags — direct evidence of "
            "volatility clustering, motivating ATR and BB Width as features."
        ),
        (
            "🔗 Correlation & Bivariate",
            "correlation_bivariate.png",
            "eda",
            "Which features are multicollinear, and which add genuinely independent information?",
            "Full correlation heatmap: Open, High, Low, Close are highly correlated (r>0.99). "
            "Volume and Daily Return show near-zero correlation with price levels. "
            "Scatter confirms volume-volatility relationship during extreme moves."
        ),
        (
            "⚙️ Technical Indicators",
            "technical_indicators.png",
            "eda",
            "Do the engineered features capture distinct market regimes?",
            "Eight-panel subplot. RSI crosses 70/30 during bull/bear runs as designed. "
            "MACD sign changes mark trend reversals. BB Width spikes precede large moves — "
            "confirming it as a volatility anticipation indicator."
        ),
        (
            "🧮 Feature Correlation (All 12)",
            "feature_correlation_full.png",
            "eda",
            "Is there redundancy in the 12-feature set?",
            "EMA-20 is highly correlated with Close (r≈0.97) — essentially a smoothed Close. "
            "ATR and BB Width moderately correlated (r≈0.6). RSI, MACD, OBV, Stoch-%K show "
            "low cross-correlation — confirming they add genuinely new information."
        ),
    ]

    for title, fname, subfolder, story_q, story_a in eda_sections:
        with st.expander(title, expanded=False):
            st.markdown(f'<div class="story-box">📖 <strong>Story:</strong> {story_q}<br><br>{story_a}</div>',
                        unsafe_allow_html=True)
            show_img(os.path.join(VIZ, subfolder, fname))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠  Model Results":

    st.markdown("# 🧠 Model Results")
    st.markdown("---")

    # ── Architecture summary ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Architectures</div>', unsafe_allow_html=True)

    arch_data = {
        "Model": ["LSTM + Attention", "GRU", "Bi-LSTM", "XGBoost"],
        "Layer 1": ["LSTM(128)", "GRU(128)", "BiLSTM(128)", "500 trees"],
        "Layer 2": ["LSTM(64)", "GRU(64)", "BiLSTM(64)", "depth=5"],
        "Key Feature": ["Temporal Attention", "Update Gate", "Bidirectional", "TreeSHAP (exact)"],
        "Input Shape": ["(60, 12)", "(60, 12)", "(60, 12)", "(720,)"],
        "Explainer": ["GradientExplainer", "—", "—", "TreeExplainer"],
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    model_sections = [
        (
            "📉 Train / Test Split",
            "train_test_split.png",
            "models",
            "Where does the model's knowledge end and its generalization begin?",
            "Time-ordered train/test boundary drawn on the full price series. Blue = train (early-mid growth); "
            "orange = test (distinct regime). Ensures true out-of-sample evaluation with no look-ahead."
        ),
        (
            "📉 LSTM Training Curves",
            "lstm_training_curves.png",
            "models",
            "Did the LSTM converge properly, or did it overfit?",
            "Full history and last-30-epoch zoom. Narrow train/val gap confirms no overfitting. "
            "EarlyStopping restores best weights. ReduceLROnPlateau drops visible as loss curve kinks."
        ),
        (
            "📊 Metrics Comparison",
            "metrics_comparison.png",
            "models",
            "Which model is best, and by how much?",
            "Three bar charts: RMSE, MAPE(%), R². Value annotations above each bar. "
            "Best model is immediately visually apparent."
        ),
        (
            "📈 All Model Predictions",
            "all_model_predictions.png",
            "models",
            "Do the models track actual price movement, or do they lag?",
            "2×2 grid, one panel per model. Actual vs predicted with shaded ±|residual| error band. "
            "RMSE and R² annotated in title. Tighter bands on predictable stretches."
        ),
        (
            "📉 Prediction Residuals",
            "prediction_errors.png",
            "models",
            "Are prediction errors random (good) or systematic (bad)?",
            "Residual bar charts per model. Random scatter around zero = well-calibrated. "
            "Systematic bias at market tops would indicate slow momentum tracking."
        ),
        (
            "🔍 Attention Matrix",
            "attention_matrix.png",
            "models",
            "Which days in the 60-day window does the LSTM actually look at?",
            "Mean attention weight bar (60 bars, one per day) + per-sample heatmap (20 samples). "
            "Model-native interpretability before SHAP. Peak at recent days = momentum follower."
        ),
    ]

    for title, fname, subfolder, story_q, story_a in model_sections:
        with st.expander(title, expanded=False):
            st.markdown(f'<div class="story-box">📖 <strong>Story:</strong> {story_q}<br><br>{story_a}</div>',
                        unsafe_allow_html=True)
            show_img(os.path.join(VIZ, subfolder, fname))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  SHAP Explainability":

    st.markdown("# 🔬 SHAP Explainability")

    st.markdown("""
    SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory.  
    It computes each feature's **marginal contribution** to a prediction by averaging over all possible feature subsets —  
    satisfying Efficiency, Symmetry, and Dummy properties that simpler methods do not.

    | Explainer | Model | Type | Samples |
    |---|---|---|---|
    | `GradientExplainer` | LSTM + Attention | Gradient-integrated | 200 BG · 50 test |
    | `TreeExplainer` | XGBoost | **Exact** Shapley values | 50 test |
    """)

    st.markdown("---")

    shap_sections = [
        (
            "🌡️ Beeswarm Summary",
            "shap_summary_beeswarm.png",
            "Which features are most impactful, and do high values push predictions up or down?",
            "Each dot = one test sample, colored by feature value (red=high, blue=low). "
            "Features ranked by mean |SHAP|. Clean gradient = monotonic relationship. "
            "Mixed colors = non-linear, context-dependent impact."
        ),
        (
            "📊 Feature Importance Bar",
            "shap_bar_importance.png",
            "What is the global feature importance leaderboard?",
            "Horizontal bar of mean |SHAP| per feature, descending. "
            "Definitive answer to: which three features does the model rely on most?"
        ),
        (
            "🌊 Waterfall Plot",
            "shap_waterfall.png",
            "How does the model arrive at one specific prediction, step by step?",
            "Each feature's contribution shown as a step from base value to final output. "
            "Fully auditable — a regulator can trace exactly why the model predicted $X."
        ),
        (
            "⚡ Force Plot",
            "shap_force_plot.png",
            "What is the push-pull dynamic between features in a single prediction?",
            "Horizontal stacked view: red = pushing prediction above baseline, "
            "blue = pulling below. Effective for non-technical stakeholder communication."
        ),
        (
            "🔵 Dependence Plots",
            "shap_dependence.png",
            "How does each indicator's value relate to its influence on the prediction?",
            "Three plots: RSI-14, MACD, ATR-14 vs their SHAP values, colored by Volume. "
            "S-shaped curve = threshold effect. Scatter = context-dependent impact."
        ),
        (
            "⏱️ Time-Step Importance",
            "shap_timestep_importance.png",
            "Does the model use recent data or historical context?",
            "Mean |SHAP| aggregated across all features per day. Peak near day 59 = momentum follower. "
            "Earlier peak = model captures longer-term structural pattern."
        ),
        (
            "🌡️ Feature × Time Heatmap (LSTM) ⭐",
            "shap_feature_time_heatmap_lstm.png",
            "Which feature contributed most, at which specific point in time?",
            "12 rows × 60 columns. Color = mean |SHAP|. Horizontal bright band = one feature dominates. "
            "Vertical bright band = one historical date was influential for all features. "
            "Global max cell outlined in blue. The centrepiece XAI visualization."
        ),
        (
            "🌡️ Feature × Time Heatmap (XGBoost)",
            "shap_feature_time_heatmap_xgb.png",
            "Does XGBoost attend to the same features and time periods as the LSTM?",
            "Same heatmap for exact TreeSHAP. Comparing both heatmaps tests consensus — "
            "if two very different model families agree, it is genuine market signal, not architecture artifact."
        ),
        (
            "🌳 XGBoost TreeSHAP Summary",
            "shap_xgb_summary.png",
            "What does a tree-based model consider important?",
            "TreeSHAP beeswarm for XGBoost. Because TreeExplainer is exact (not approximated), "
            "this is ground-truth SHAP for the tree model. Compare with LSTM GradientExplainer ranking."
        ),
        (
            "⚖️ LSTM vs XGBoost Comparison",
            "shap_model_comparison.png",
            "Where do deep learning and gradient boosting agree — and disagree?",
            "Grouped bar chart per feature. Both bars tall = genuine signal both families detect. "
            "Only one tall = model-specific reliance — potentially architecture-specific inductive bias."
        ),
    ]

    for title, fname, story_q, story_a in shap_sections:
        with st.expander(title, expanded=False):
            st.markdown(f'<div class="story-box">📖 <strong>Story:</strong> {story_q}<br><br>{story_a}</div>',
                        unsafe_allow_html=True)
            show_img(os.path.join(VIZ, "shap", fname))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🖥️  Final Dashboard":

    st.markdown("# 🖥️ Final Dashboard")
    st.markdown("Interactive Plotly dashboards generated by the notebook.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 EDA Dashboard", "📈 Final Model Dashboard"])

    with tab1:
        st.markdown('<div class="section-header">EDA Dashboard</div>', unsafe_allow_html=True)
        eda_html = os.path.join(VIZ, "dashboard", "eda_dashboard.html")
        if os.path.exists(eda_html):
            with open(eda_html, "r") as f:
                st.components.v1.html(f.read(), height=1050, scrolling=True)
        else:
            st.info("Run the notebook to generate `visualizations/dashboard/eda_dashboard.html`")

    with tab2:
        st.markdown('<div class="section-header">Final Model & XAI Dashboard</div>', unsafe_allow_html=True)
        final_html = os.path.join(VIZ, "dashboard", "final_dashboard.html")
        if os.path.exists(final_html):
            with open(final_html, "r") as f:
                st.components.v1.html(f.read(), height=1650, scrolling=True)
        else:
            st.info("Run the notebook to generate `visualizations/dashboard/final_dashboard.html`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄  Reports":

    st.markdown("# 📄 Auto-Generated Reports")
    st.markdown("All reports are saved to `reports/` by the notebook at runtime.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 Initial Summary", "📊 KPI Summary", "📝 Final Summary"])

    with tab1:
        st.markdown('<div class="section-header">initial_summary.txt</div>', unsafe_allow_html=True)
        st.markdown("Generated at the start of the pipeline — raw dataset shape, NaN counts, dtypes, and descriptive statistics before any cleaning.")
        st.code(read_txt(RPT_INITIAL), language="text")

    with tab2:
        st.markdown('<div class="section-header">kpi_summary.txt</div>', unsafe_allow_html=True)
        st.markdown("Generated after all models are trained — best model KPIs, all metrics table, top SHAP features, and dataset statistics.")
        st.code(read_txt(RPT_KPI), language="text")

    with tab3:
        st.markdown('<div class="section-header">final_summary.txt</div>', unsafe_allow_html=True)
        st.markdown("Complete project narrative — methodology, results, XAI insights, and output file index.")
        st.code(read_txt(RPT_FINAL), language="text")

    st.markdown("---")
    st.markdown("**Download Reports**")
    for label, path in [
        ("📥 initial_summary.txt", RPT_INITIAL),
        ("📥 kpi_summary.txt",     RPT_KPI),
        ("📥 final_summary.txt",   RPT_FINAL),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
            st.download_button(label, data=content, file_name=os.path.basename(path),
                               mime="text/plain")
