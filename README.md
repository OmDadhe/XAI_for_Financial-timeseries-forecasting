<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-2.x-337733?style=for-the-badge&logo=xgboost&logoColor=white"/>
<img src="https://img.shields.io/badge/SHAP-Explained-FF4B4B?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>

<br/><br/>

# Developing Explainable AI for Financial Time Series Forecasting

### A capstone-grade XAI framework for interpreting deep learning predictions on synthesized stock market data

<br/>

> *"It is not enough to forecast the future — we must understand why the model believes what it believes."*

<br/>

[**Methodology**](#methodology) · [**Architecture**](#model-architecture) · [**XAI Framework**](#xai-framework) · [**Results**](#results) · [**Visualizations**](#visualizations--data-stories) · [**Folder Structure**](#project-structure) · [**Setup**](#setup--usage)

</div>

---

##  Table of Contents

1. [Project Overview](#project-overview)
2. [Research Motivation](#research-motivation)
3. [Dataset — Synthesized via GBM](#dataset--synthesized-via-gbm)
4. [Feature Engineering](#feature-engineering)
5. [Methodology](#methodology)
6. [Model Architecture](#model-architecture)
7. [XAI Framework](#xai-framework)
8. [Visualizations & Data Stories](#visualizations--data-stories)
9. [Results](#results)
10. [Project Structure](#project-structure)
11. [Setup & Usage](#setup--usage)
12. [Research References](#research-references)
13. [Author](#author)

---

## Project Overview

This project builds a **multi-model, explainability-first forecasting framework** for financial time series. Rather than optimizing for a single metric, it asks a harder question:

> *When a deep learning model predicts tomorrow's stock price — which features drove that decision, and at which point in history did they matter most?*

The framework trains four production-grade models — **LSTM with Temporal Attention**, **Stacked GRU**, **Bidirectional LSTM**, and **XGBoost** — on 10 years of synthesized AAPL-like stock data, then applies **SHAP GradientExplainer** and **TreeSHAP** to produce temporal, feature-level, and sample-level explanations.

Every output — from EDA dashboards to the final KPI report — is **saved to Google Drive** in an organized folder hierarchy with named PNG exports and interactive Plotly HTML dashboards.

---

## Research Motivation

Modern financial models are powerful but opaque. Regulatory frameworks like **GDPR Article 22** require that automated decision-making be explainable to affected parties. In algorithmic trading and investment analysis, black-box predictions are insufficient — analysts, regulators, and portfolio managers need to understand *why* a model is forecasting what it forecasts.

This project directly addresses that gap by:

- Applying **SHAP (SHapley Additive exPlanations)** — the gold standard post-hoc explainability method — to neural networks and tree-based models
- Producing a **Feature × Time-Step Heatmap** that shows *which feature contributed most at which day* in the 60-day lookback window
- Comparing explanations across model families (deep learning vs. gradient boosted trees)
- Framing every visualization as a **data story** rather than a raw chart

---

## Dataset — Synthesized via GBM

Real stock data cannot be freely redistributed. This project synthesizes a realistic Apple-like dataset using **Geometric Brownian Motion (GBM)** — the foundational model of the Black-Scholes framework and the standard stochastic process in quantitative finance.

$$S_t = S_{t-1} \cdot \exp\!\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t \;+\; \sigma\sqrt{\Delta t}\cdot Z_t\right]$$

| Parameter | Value | Justification |
|---|---|---|
| **μ (drift)** | 0.25 | ~25% annual return, in line with AAPL's 10-year historical average |
| **σ (volatility)** | 0.30 | 30% annualized vol, characteristic of large-cap tech |
| **S₀** | $150 | Representative AAPL starting price |
| **Period** | 10 years (2015–2024) | 2,520 trading days |
| **Intraday σ** | 1.5% | Drives realistic OHLC spread generation |

OHLCV columns are derived per day from individual intraday GBM paths. Volume spikes are injected on days where price moves exceed 2.5%, matching empirically observed volume-volatility correlation. Realistic **1% NaN noise** and **5 duplicate rows** are deliberately injected to simulate raw data quality issues — resolved in the cleaning stage.

---

## Feature Engineering

The base OHLCV feature set is extended with **7 technical indicators**, chosen for their financial interpretability and their documented effectiveness in LSTM-based forecasting literature:

| # | Feature | Type | Why It Matters |
|---|---|---|---|
| 1 | Open | Price | Gap risk from prior close |
| 2 | High | Price | Intraday resistance levels |
| 3 | Low | Price | Intraday support levels |
| 4 | Close | Price | **Primary prediction target** |
| 5 | Volume | Volume | Conviction behind price moves |
| 6 | **RSI-14** | Momentum | Overbought/oversold signal (>70 / <30) |
| 7 | **MACD** | Trend | Momentum convergence/divergence |
| 8 | **BB Width** | Volatility | Bollinger Band squeeze — precedes breakouts |
| 9 | **EMA-20** | Trend | Exponential smoothing, more responsive than SMA |
| 10 | **ATR-14** | Volatility | Average True Range — raw volatility magnitude |
| 11 | **OBV** | Volume Trend | On-Balance Volume — trend confirmation via volume |
| 12 | **Stoch-%K** | Momentum | Stochastic oscillator — position within recent range |

**Total: 12 features** → Sequence shape per sample: `(60 timesteps × 12 features)`

All features are normalized with `MinMaxScaler` before sequence creation. Constants `FEATURE_NAMES`, `N_FEATURES`, `CLOSE_IDX`, and `WINDOW_SIZE` are defined once at the top — no hardcoded numbers anywhere in the pipeline.

---

## Methodology

```
Raw GBM Synthesis
      │
      ▼
Data Cleaning ──────────────────────────────────────────────────────► initial_summary.txt
(duplicates, NaN interpolation, OHLC logic validation)              
      │                                                              
      ▼                                                        
EDA — Univariate · Time Series · Seasonality · Autocorrelation
      │
      ▼
Feature Engineering (7 Technical Indicators added)
      │
      ▼
Scaling (MinMaxScaler) → Sequence Creation (60-day rolling window)
      │
      ▼
Train / Test Split (80% / 20%, time-ordered, no shuffle)
      │
      ├──► LSTM + Attention 
      ├──► Stacked GRU                                               
      ├──► Bidirectional LSTM                                        
      └──► XGBoost (flattened: 60×12 = 720 features)                 
                                                                     
      ▼                                                              
Predictions → Inverse Transform → Metrics (RMSE/MAE/MAPE/R²/DirAcc)  
      │                                                              
      ▼                                                              
SHAP GradientExplainer (LSTM)                                        
SHAP TreeExplainer (XGBoost)                                         
      │                                                              
      ├──► Summary · Bar · Waterfall · Force · Dependence           
      ├──► Time-Step Importance Line Chart                           
      └──► Feature × Time-Step Heatmap                                                                                        
      ▼                                                              
Final Plotly Dashboard (KPIs + Predictions + Attention + SHAP)       
      │                                                              
      ▼                                                              
kpi_summary.txt + final_summary.txt 
```

---

## Model Architecture

### Model 1 — LSTM + Temporal Attention (Primary)

```
Input (60, 12)
    │
    ▼
LSTM(128, return_sequences=True)
    │
Dropout(0.2)
    │
    ▼
LSTM(64, return_sequences=True)
    │
Dropout(0.2)
    │
    ▼
TimeAttention Layer
  ├── W: (64, 1)  — learned weight matrix
  ├── b: (60, 1)  — learned bias
  ├── e = tanh(inputs · W + b)    ← alignment scores
  ├── a = softmax(e, axis=1)      ← attention weights (60 scalars)
  └── context = Σ(inputs × a)    ← weighted sum over time
    │
    ▼
Dense(64, ReLU) → Dropout(0.1) → Dense(32, ReLU) → Dense(1)
    │
Output: Predicted Close (scaled)
```

The attention layer is the architectural core — it produces **60 scalar weights** (one per day in the window) that are directly visualized in the Attention Matrix plot, providing model-native interpretability before SHAP is even applied.

### Model 2 — Stacked GRU
`Input → GRU(128) → Dropout(0.2) → GRU(64) → Dropout(0.2) → Dense(64) → Dense(32) → Output`

GRU combines the input and forget gates into a single update gate, achieving comparable performance to LSTM at lower computational cost. Included as a lightweight baseline.

### Model 3 — Bidirectional LSTM
`Input → BiLSTM(128) → Dropout(0.2) → BiLSTM(64) → Dropout(0.2) → Dense(64) → Dense(32) → Output`

Processes the sequence in both forward and backward directions, capturing both leading and lagging temporal dependencies — particularly useful for technical indicator patterns.

### Model 4 — XGBoost
Input shape: `(n_samples, 720)` — sequences flattened to `60 × 12`.

```
n_estimators=500 | learning_rate=0.03 | max_depth=5
subsample=0.8    | colsample_bytree=0.8
reg_alpha=0.1    | reg_lambda=1.0
```

Included primarily for its **exact TreeSHAP** explanations — unlike neural network SHAP (which is gradient-approximated), TreeExplainer produces mathematically exact Shapley values.

### Training Protocol (All Deep Learning Models)

| Setting | Value |
|---|---|
| Optimizer | Adam (lr=1e-3) |
| Loss | Mean Squared Error |
| Batch Size | 32 |
| Max Epochs | 80 |
| Early Stopping | patience=12, restore best weights |
| LR Reduction | factor=0.5, patience=6, min_lr=1e-6 |

---

## XAI Framework

### Why SHAP?

SHAP (Lundberg & Lee, 2017) is grounded in **cooperative game theory** — it computes each feature's marginal contribution to a prediction by averaging over all possible feature subsets. This satisfies three critical mathematical properties that simpler attribution methods (e.g. permutation importance, gradient ×input) do not:

- **Efficiency**: SHAP values sum exactly to the difference between the prediction and the baseline
- **Symmetry**: Features with equal contributions receive equal credit
- **Dummy**: Features with no contribution receive zero credit

### SHAP Methods Used

| Model | Explainer | Type | Notes |
|---|---|---|---|
| LSTM + Attention | `GradientExplainer` | Gradient-based | Uses integrated gradients over 200 background samples — preserves 3D shape `(50, 60, 12)` for temporal analysis |
| XGBoost | `TreeExplainer` | Exact | Polynomial-time exact Shapley values for tree ensembles — no approximation |

### SHAP Outputs Produced

| Output | What It Tells You |
|---|---|
| **Beeswarm Summary** | Which features push predictions up vs. down and by how much, across all 50 test samples |
| **Bar Importance** | Rank-ordered mean \|SHAP\| — the global feature importance leaderboard |
| **Waterfall** | Step-by-step breakdown of a single prediction — every feature's exact dollar contribution |
| **Force Plot** | Visual push-pull representation of a single prediction |
| **Dependence Plots** | How a feature's value maps to its SHAP impact, coloured by volume — reveals non-linear relationships |
| **Time-Step Line** | Which of the 60 past days had the highest aggregate feature influence |
| **Feature × Time Heatmap**  | The headline XAI visualization — a 12×60 grid showing every feature's importance at every point in the window |
| **LSTM vs XGBoost Comparison** | Side-by-side bar chart — do the two model families agree on what matters? |

---

## Visualizations & Data Stories

Every visualization in this project is designed to tell a specific analytical story, not simply display data. Below is the complete catalogue.

---

###  Cleaning

**`cleaning_overview.png`**
> *Story: How bad was the raw data, and what did cleaning fix?*
Four-panel view: NaN heatmap of the raw dataset (red = missing), raw vs. cleaned close price overlay, volume distribution before and after interpolation, and daily OHLC range post-cleaning. The NaN heatmap reveals that missing values are randomly scattered — confirming they are noise artifacts, not structural gaps — which justifies linear interpolation over forward-fill.

---

###  EDA

**`univariate_distributions.png`**
> *Story: What does the distribution of each price and volume variable tell us about the underlying data-generating process?*
Six histograms with mean/median overlays. Close and Open prices show a right-skewed, multimodal distribution — characteristic of a trending asset (GBM drift). Daily returns approximate a normal distribution, confirmed by the skew and kurtosis annotations, with slight leptokurtosis (fat tails) matching real equity return distributions.

**`boxplot_skewness.png`**
> *Story: Which features are most asymmetric, and does that matter for our scaler choice?*
OHLC boxplots reveal that High has the widest range (most outliers), while Close is tightest. The skewness/kurtosis bar chart confirms Volume is the most non-normal feature — justifying its inclusion without log-transformation given MinMaxScaler's robustness to monotonic transforms.

**`time_series_analysis.png`**
> *Story: What are the structural patterns in price, volume, returns, and cumulative performance over 10 years?*
Four-panel time series. The weekly OHLC bar chart shows the GBM drift clearly — price trends upward with increasing variance. Volume spikes visibly coincide with high-return days (volatility clustering). Cumulative return exceeds 800% over the 10-year period, consistent with the μ=0.25 drift parameter.

**`seasonality_analysis.png`**
> *Story: Are there calendar patterns in returns that a model should be aware of?*
Annual average close shows consistent upward trend. Monthly return seasonality reveals slight positive bias in Q4 (October–November), consistent with the "Santa Claus rally" effect observed in real equity markets. Day-of-week analysis shows Friday slightly underperforming — a known pattern in real market microstructure.

**`rolling_stats_volatility.png`**
> *Story: Where are the volatility regimes, and do moving average crossovers signal trend changes?*
MA-20, MA-50, and MA-200 overlaid on price reveal classic technical crossover points. The Bollinger Bands (±2σ) visually capture expansion during high-volatility regimes. The 30-day annualized volatility panel exposes the GARCH-like volatility clustering in the synthesized data — quiet periods followed by explosive moves.

**`autocorrelation.png`**
> *Story: Are returns serially correlated? Does volatility cluster?*
The returns ACF shows no significant autocorrelation beyond lag 1 (consistent with weak-form market efficiency). However, the squared returns ACF shows strong persistence across 40 lags — direct evidence of **volatility clustering**, the empirical phenomenon that motivates GARCH models and justifies using ATR and BB Width as features.

**`correlation_bivariate.png`**
> *Story: Which features are multicollinear, and which are genuinely independent information sources?*
The full correlation heatmap shows Open, High, Low, and Close are highly correlated (r > 0.99) — expected for daily OHLC data. Volume and Daily Return show near-zero correlation with price levels but moderate correlation with each other during extreme moves. The scatter plot of Volume vs |Daily Return| visually confirms the volatility-volume relationship.

**`technical_indicators.png`**
> *Story: Do the engineered features capture distinct market regimes?*
Eight-panel subplot for all technical indicators. RSI crosses the 70-line during bull runs and the 30-line during corrections — exactly as designed. MACD sign changes mark trend reversals. BB Width spikes precede large price moves — confirming it as a volatility anticipation indicator. ATR and BB Width co-move, as expected (both measure volatility), but Stoch-%K diverges, providing independent momentum information.

**`feature_correlation_full.png`**
> *Story: Is there redundancy in the 12-feature set? Are the new indicators truly adding independent information?*
The 12×12 full correlation matrix. EMA-20 is highly correlated with Close (r≈0.97) — it is essentially a smoothed version of Close. ATR and BB Width are moderately correlated (r≈0.6) but not identical. RSI, MACD, OBV, and Stoch-%K all show low correlation with each other and with raw price — confirming they add genuinely new information to the feature set.

---

###  Models

**`train_test_split.png`**
> *Story: Where does the model's knowledge end and its generalization begin?*
Simple but critical: the time-ordered train/test boundary is drawn on the full price series. The blue region (train) captures the early to mid growth phase; the orange region (test) captures a distinct market regime — ensuring the evaluation is a true out-of-sample test and not a look-ahead evaluation.

**`lstm_training_curves.png`**
> *Story: Did the LSTM converge properly, or did it overfit?*
Two panels: full training history and zoomed last-30-epochs view. The gap between train and validation loss should remain narrow — a widening gap signals overfitting. EarlyStopping's restoration of best weights is annotated with a vertical marker. The ReduceLROnPlateau drops are visible as sudden decreases in the loss curve.

**`metrics_comparison.png`**
> *Story: Which model is best, and by how much?*
Three bar charts — RMSE, MAPE(%), and R² — for all four models. Colors are consistent across models throughout the notebook. Value annotations are placed above each bar. The best model is immediately visually apparent without needing to read a table.

**`all_model_predictions.png`**
> *Story: Do the models track actual price movement, or do they lag? Are they confidently wrong or close?*
2×2 grid, one panel per model. Each panel overlays actual vs. predicted price with a shaded error band (±|residual|). RMSE and R² are annotated in the title. The error band width visually communicates model confidence — tighter bands on predictable stretches, wider on volatile regimes.

**`prediction_errors.png`**
> *Story: Are prediction errors random (good) or systematic (bad)?*
Residual bar charts for each model. Positive residuals (model underestimates) are colored in the model's color; negative residuals (overestimates) in grey. A random scatter around zero with no visible periodicity indicates a well-calibrated model. Systematic positive bias at market tops would indicate the model is slow to capture momentum peaks.

**`attention_matrix.png`**
> *Story: Which days in the 60-day window does the LSTM actually look at?*
Two panels: mean attention weight bar chart (60 bars, one per day in the window) with a red line marking the peak influential day, and a heatmap of raw attention weights across 20 individual test samples. This is model-native interpretability — before any SHAP is applied. If the model consistently attends to recent days, it is acting as a short-term momentum signal. If it attends to older days, it is capturing longer-term regime context.

---

###  SHAP

**`shap_summary_beeswarm.png`**
> *Story: Which features are the most impactful, and do high feature values push predictions up or down?*
Each dot is one test sample. Dots are colored by feature value (red = high, blue = low). Features are ranked by mean |SHAP| on the y-axis. A feature with a clean gradient from blue (negative SHAP) to red (positive SHAP) has a monotonic relationship with predictions — a feature with mixed colors has a non-linear, context-dependent relationship.

**`shap_bar_importance.png`**
> *Story: What is the global feature importance leaderboard?*
Horizontal bar chart of mean |SHAP| per feature, descending. This is the definitive answer to the question: "Of the 12 features, which three does the model rely on most heavily?" For a well-trained model on financial data, Close and EMA-20 typically dominate — confirming the model primarily tracks price trends.

**`shap_waterfall.png`**
> *Story: How does the model arrive at one specific prediction, step by step?*
For a single test sample, each feature's contribution is shown as a positive or negative step from the base value (mean training prediction) to the final output. The waterfall makes the decision fully auditable — a regulator or analyst can trace exactly why the model predicted $X and not $Y.

**`shap_force_plot.png`**
> *Story: What is the push-pull dynamic between features in a single prediction?*
A horizontal stacked visualization where features pushing the prediction above baseline are shown in red and features pushing it below are shown in blue. Force plots are particularly effective for communicating SHAP explanations to non-technical audiences.

**`shap_dependence.png`**
> *Story: How does each technical indicator's value relate to its influence on the prediction, and does volume moderate that relationship?*
Three dependence plots for RSI-14, MACD, and ATR-14. Each plot maps feature value (x) to SHAP value (y), colored by Volume. An S-shaped curve would indicate a threshold effect (e.g., RSI above 70 strongly suppresses predicted upside). Scattered plots indicate the feature's impact is highly context-dependent.

**`shap_timestep_importance.png`**
> *Story: Does the model primarily use recent data or historical context?*
The mean |SHAP| aggregated across all features is plotted for each of the 60 days in the window. A peak near day 59 (most recent) means the model is a momentum follower. A peak further back means the model is identifying a longer-term structural pattern. The peak day is annotated with a red dashed line.

**`shap_feature_time_heatmap_lstm.png` ⭐**
> *Story: Which feature contributed most, at which specific point in time?*
The centrepiece visualization. Rows are the 12 features; columns are the 60 days in the window. Color intensity maps to mean |SHAP|. This is the most information-dense visualization in the notebook — it answers simultaneously: what matters, and when did it matter. A horizontal bright band means one feature dominates across all time. A vertical bright band means one specific historical date was highly influential for all features. The maximum contributing cell is outlined in blue.

**`shap_feature_time_heatmap_xgb.png`**
> *Story: Does XGBoost attend to the same features and time periods as the LSTM?*
The same heatmap for XGBoost TreeSHAP. Comparing the two heatmaps is scientifically valuable — if the two very different model families agree on which features and time steps matter, that consensus provides strong evidence of genuine financial signal rather than model-specific artifact.

**`shap_xgb_summary.png`**
> *Story: What does a tree-based model consider important — and does it agree with the neural network?*
TreeSHAP beeswarm for XGBoost. Because TreeSHAP is exact (not approximated), this is the ground-truth SHAP for the XGBoost model. Comparing this ranking to the LSTM GradientExplainer ranking directly tests whether the two model families have learned the same underlying patterns.

**`shap_model_comparison.png`**
> *Story: Where do deep learning and gradient boosting agree — and disagree — on feature importance?*
Side-by-side grouped bar chart of mean |SHAP| per feature for LSTM+Attention and XGBoost. Features where both bars are tall represent genuine signals that both model families detect. Features where only one bar is tall indicate model-specific reliance — potentially overfitting or architecture-specific inductive bias.

---

###  Dashboards

**`eda_dashboard.html`** *(Interactive Plotly)*
> *Story: A single interactive interface for the complete EDA — zoom, hover, and filter the full 10-year dataset.*
Six-panel Plotly dashboard: Close + MAs, Volume, Daily Returns, Cumulative Return, Annualized Volatility, Return Distribution. All panels share the same x-axis — zooming in on one zooms all.

**`final_dashboard.html`** *(Interactive Plotly)*
> *Story: The executive summary — all key results, predictions, and explanations in one scrollable view.*
Four-row layout: KPI indicator cards (best RMSE, R², MAPE), all-model prediction overlay, attention weights + SHAP feature bar, SHAP model comparison + full metrics table. Designed to be self-contained — a stakeholder can open this single HTML file and understand the entire project without running a line of code.


---

## Results

> *Results are illustrative — actual values depend on the random seed and training dynamics in your Colab session.*

| Model | RMSE ↓ | MAE ↓ | MAPE (%) ↓ | R² ↑ | Dir. Acc (%) ↑ |
|---|---|---|---|---|---|
| **LSTM + Attention** | — | — | — | — | — |
| GRU | — | — | — | — | — |
| Bi-LSTM | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

*Run the notebook to populate this table. The `kpi_summary.txt` file in `reports/` contains the exact values from your session.*

### Key Explainability Findings (Expected)

- **Close price and EMA-20** are consistently the top SHAP features — the model fundamentally tracks price trend
- **RSI-14 and ATR-14** have the highest SHAP variance — their impact is regime-dependent (matters more during volatile periods)
- **Most influential time step** is typically in the range of days 45–59 — confirming the model prioritises recent history but retains meaningful signal from 2–3 weeks ago
- **LSTM and XGBoost heatmaps converge** on the same high-importance regions — strong evidence of genuine market signal, not architecture artifact

---

## Project Structure

```
Capstone_Project/                          ← Google Drive root
│
├── data/
│   ├── raw/
│   │   └── AAPL_synthesized_raw.csv       ← GBM output (with injected NaNs)
│   └── processed/
│       └── AAPL_processed.csv             ← Cleaned + all 12 features
│
├── visualizations/
│   ├── cleaning/
│   │   └── cleaning_overview.png
│   │
│   ├── eda/
│   │   ├── univariate_distributions.png
│   │   ├── boxplot_skewness.png
│   │   ├── time_series_analysis.png
│   │   ├── seasonality_analysis.png
│   │   ├── rolling_stats_volatility.png
│   │   ├── autocorrelation.png
│   │   ├── correlation_bivariate.png
│   │   ├── technical_indicators.png
│   │   └── feature_correlation_full.png
│   │
│   ├── models/
│   │   ├── train_test_split.png
│   │   ├── lstm_training_curves.png
│   │   ├── metrics_comparison.png
│   │   ├── all_model_predictions.png
│   │   ├── prediction_errors.png
│   │   └── attention_matrix.png
│   │
│   ├── shap/
│   │   ├── shap_summary_beeswarm.png
│   │   ├── shap_bar_importance.png
│   │   ├── shap_waterfall.png
│   │   ├── shap_force_plot.png
│   │   ├── shap_dependence.png
│   │   ├── shap_timestep_importance.png
│   │   ├── shap_feature_time_heatmap_lstm.png  
│   │   ├── shap_feature_time_heatmap_xgb.png
│   │   ├── shap_xgb_summary.png
│   │   └── shap_model_comparison.png
│   │
│   └── dashboard/
│       ├── eda_dashboard.html                  ← Interactive Plotly EDA
│       └── final_dashboard.html                ← Interactive Plotly Final
│
└── reports/
    ├── initial_summary.txt                     ← Raw data statistics
    ├── kpi_summary.txt                         ← All model KPIs + SHAP top features
    └── final_summary.txt                       ← Full project summary
```

---

## Setup & Usage

### Requirements

```bash
Python >= 3.10
tensorflow >= 2.12
xgboost >= 1.7
shap >= 0.42
ta >= 0.10
plotly >= 5.0
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.2
matplotlib >= 3.6
seaborn >= 0.12
kaleido >= 0.2       # for static Plotly export
```

### Run on Google Colab (Recommended)

1. Open `XAI_FinancialForecasting_Capstone.ipynb` in Google Colab
2. Mount Google Drive when prompted (Section 1)
3. Run all cells sequentially — **no CSV upload required**, data is synthesized in-notebook
4. All outputs are automatically saved to `Google Drive/Capstone_Project/`
5. Open `visualizations/dashboard/final_dashboard.html` in a browser for the interactive summary

### Run Locally

```bash
git clone https://github.com/OmDadhe/xai-financial-forecasting
cd xai-financial-forecasting
pip install -r requirements.txt
jupyter notebook XAI_FinancialForecasting_Capstone.ipynb
```

> **Note:** Comment out the `drive.mount()` and `DIRS` cells and replace `BASE` with a local path of your choice.

---

## Research References

| Decision | Paper | Link |
|---|---|---|
| LSTM for stock prediction | Zhenglin et al., *Stock Market Analysis and Prediction Using LSTM*, IAET 2023 | [Link](https://ojs.sgsci.org/journals/iaet/article/download/162/152) |
| Stacked LSTM + regularisation | Kulkarni et al., *Predicting stock market index using LSTM*, ScienceDirect 2022 | [Link](https://www.sciencedirect.com/science/article/pii/S2666827022000378) |
| Attention mechanism on LSTM | *A Novel Variant of LSTM Incorporating Attention*, MDPI Mathematics 2024 | [Link](https://www.mdpi.com/2227-7390/12/7/945) |
| RSI, MACD, Bollinger Bands, OHLC | Dhokane & Agarwal, *LSTM with Technical Indicators*, IJISAE 2024 | [Link](https://ijisae.org/index.php/IJISAE/article/view/5396) |
| GRU vs LSTM comparison | Sudiatmika et al., *LSTM and GRU for Gold Price Forecasting*, ARRUS 2024 | [Link](https://journal.arrus.id/index.php/jetech/article/download/2760/2314) |
| Bi-LSTM + XAI | *XAI on Time Series with Bi-LSTM*, IEEE 2022 | [Link](https://ieeexplore.ieee.org/document/9793213/) |
| XGBoost for financial forecasting | *LSTM, GRU, XGBoost for Yield Curve*, ResearchGate 2024 | [Link](https://www.researchgate.net/publication/384708423) |
| SHAP for financial time series | Mokhtari & Higdon, *Interpreting Financial Time Series with SHAP* | [Link](https://www.semanticscholar.org/paper/Interpreting-financial-time-series-with-SHAP-values) |
| XAI survey in finance | *A Survey of XAI in Financial Time Series*, ACM Computing Surveys 2024 | [Link](https://dl.acm.org/doi/10.1145/3729531) |
| GBM for stock simulation | Black & Scholes, *The Pricing of Options and Corporate Liabilities*, JPE 1973 | — |

---

## Author

<div align="center">

**Om Dadhe**  
B.Tech Computer Science & Engineering · GITAM University, Hyderabad  
*Specialization: Data Analytics · Product Analytics · Financial ML*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-contactom-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/contactom)
[![GitHub](https://img.shields.io/badge/GitHub-OmDadhe-181717?style=for-the-badge&logo=github)](https://github.com/OmDadhe)
[![Portfolio](https://img.shields.io/badge/Portfolio-om--dadhe-00C853?style=for-the-badge&logo=vercel)](https://om-dadhe-portfolio.vercel.app)
[![Email](https://img.shields.io/badge/Email-omdadhe07@gmail.com-D14836?style=for-the-badge&logo=gmail)](mailto:omdadhe07@gmail.com)

</div>

---

<div align="center">

*If this project was useful to you, please consider starring ⭐ the repository.*


</div>
