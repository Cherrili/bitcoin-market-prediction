"""
Bitcoin Market State Prediction — Streamlit Dashboard

Run:
    python -m streamlit run app.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import streamlit as st

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

MODEL_NAMES = [
    "Logistic_Regression",
    "Random_Forest",
    "XGBoost",
    "LightGBM",
    "SVM",
    "KNN",
    "Voting_Ensemble",
]
MODEL_DISPLAY = {m: m.replace("_", " ") for m in MODEL_NAMES}

FEAT_IMP_FILES = {
    "Random Forest":             "feature_importance_rf.png",
    "XGBoost":                   "feature_importance_xgb.png",
    "LightGBM":                  "feature_importance_lgbm.png",
    "Logistic Regression":       "feature_importance_lr.png",
    "Combined (RF + XGB + LGB)": "feature_importance_combined.png",
}

ACCENT   = "#F7931A"   # Bitcoin orange
BG_CARD  = "#1E1E2E"
BG_PAGE  = "#0F0F1A"


# ── page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Bitcoin Market Prediction",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* ── global background & base text ── */
  .stApp {{ background-color: {BG_PAGE}; }}
  .stApp p, .stApp li, .stApp span,
  .stApp div {{ color: #D8D8EE; }}

  /* ── sidebar ── */
  [data-testid="stSidebar"] {{
      background-color: {BG_CARD};
      border-right: 1px solid #33334A;
  }}
  [data-testid="stSidebar"] * {{ color: #D8D8EE !important; }}

  /* ── metric cards ── */
  [data-testid="stMetric"] {{
      background: {BG_CARD};
      border: 1px solid #33334A;
      border-radius: 10px;
      padding: 16px 20px;
  }}
  [data-testid="stMetricLabel"] {{ color: #B0B0CC !important; font-size: 0.82rem; }}
  [data-testid="stMetricValue"] {{ color: {ACCENT} !important; font-size: 1.6rem; font-weight: 700; }}

  /* ── section headers ── */
  h1 {{ color: {ACCENT} !important; letter-spacing: -0.5px; }}
  h2, h3 {{ color: #EEEEFF !important; }}

  /* ── markdown tables ── */
  .stMarkdown table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
  }}
  .stMarkdown table th {{
      color: #B0B0CC !important;
      background: #22223A !important;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 10px 14px;
      border-bottom: 2px solid #33334A;
  }}
  .stMarkdown table td {{
      color: #E0E0F4 !important;
      padding: 9px 14px;
      border-bottom: 1px solid #2A2A40;
      background: transparent !important;
  }}
  .stMarkdown table tr:hover td {{
      background: #1E1E36 !important;
  }}
  .stMarkdown code {{
      color: {ACCENT} !important;
      background: #22223A !important;
      padding: 2px 6px;
      border-radius: 4px;
  }}

  /* ── divider ── */
  hr {{ border-color: #33334A !important; }}

  /* ── dataframe ── */
  [data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}
  [data-testid="stDataFrame"] iframe {{ color-scheme: dark; }}

  /* ── tabs ── */
  .stTabs [data-baseweb="tab"] {{ color: #B0B0CC !important; }}
  .stTabs [data-baseweb="tab"][aria-selected="true"] {{
      border-bottom: 3px solid {ACCENT};
      color: {ACCENT} !important;
  }}

  /* ── caption ── */
  .stCaption {{ color: #9898B8 !important; }}

  /* ── selectbox label ── */
  label {{ color: #C0C0D8 !important; }}

  /* ── alert ── */
  .stAlert {{ border-radius: 8px; }}

  /* ── selectbox ── */
  [data-baseweb="select"] > div {{
      background-color: #22223A !important;
      border: 1px solid #44445A !important;
      border-radius: 8px !important;
  }}
  [data-baseweb="select"] span,
  [data-baseweb="select"] div {{
      color: #E0E0F4 !important;
  }}
  [data-baseweb="popover"] [role="option"] {{
      background-color: #22223A !important;
      color: #E0E0F4 !important;
  }}
  [data-baseweb="popover"] [role="option"]:hover {{
      background-color: #33334A !important;
  }}
  [data-baseweb="popover"] [aria-selected="true"] {{
      background-color: #F7931A22 !important;
      color: {ACCENT} !important;
  }}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def output_ready() -> bool:
    return os.path.exists(os.path.join(OUTPUT_DIR, "model_summary.csv"))


def load_image(filename: str) -> Image.Image:
    return Image.open(os.path.join(OUTPUT_DIR, filename))


def load_summary() -> pd.DataFrame:
    return pd.read_csv(os.path.join(OUTPUT_DIR, "model_summary.csv"))


def badge(text: str, color: str = ACCENT) -> str:
    return (f'<span style="background:{color};color:#000;padding:2px 9px;'
            f'border-radius:12px;font-size:0.78rem;font-weight:600;">{text}</span>')


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"## ₿ BTC Prediction")
    st.caption("On-Chain ML Dashboard")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Overview", "Model Performance", "Confusion Matrix",
         "ROC Curves", "Feature Importance", "LSTM", "RL Agent (DQN)"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Dataset: Kaggle Bitcoin On-Chain Data  \n"
               "Period: Aug 2010 — Sep 2023  \n"
               "Split: 80 / 20 (time-ordered)")

# ── guard ─────────────────────────────────────────────────────────────────────

if not output_ready():
    st.error("Output files not found. Please run training first:  \n"
             "`python -m src.main`")
    st.stop()


# ── pages ─────────────────────────────────────────────────────────────────────

def page_overview():
    st.title("₿ Bitcoin Market State Prediction")
    st.caption("Predicting Bull / Sideways / Bear market regimes using on-chain blockchain indicators")
    st.divider()

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("Project Summary")
        st.markdown(f"""
| Field | Detail |
|-------|--------|
| **Dataset** | Bitcoin Network On-Chain Data (Kaggle) |
| **Task** | 3-class market state classification |
| **Label** | 30-day forward price return |
| **Bull** {badge("▲ > +15%", "#28a745")} | Label = 1 |
| **Bear** {badge("▼ < −15%", "#dc3545")} | Label = −1 |
| **Sideways** {badge("±15%", "#fd7e14")} | Label = 0 |
| **Date range** | Sep 2010 → Aug 2023 |
| **Features** | 176 (raw + rolling + lag + momentum + MVRV) |
| **Train / Test** | 80 / 20 — strict time order, no shuffle |
""", unsafe_allow_html=True)

    with col_right:
        st.subheader("Models & Class Balancing")
        st.markdown(f"""
| Model | Tuning | Class Balancing |
|-------|--------|-----------------|
| Logistic Regression | GridSearchCV | `class_weight="balanced"` |
| Random Forest | GridSearchCV | `class_weight="balanced"` |
| XGBoost | GridSearchCV | `sample_weight` |
| LightGBM | GridSearchCV | `class_weight="balanced"` |
| SVM | GridSearchCV | `class_weight="balanced"` |
| KNN | GridSearchCV | — |
| **Voting Ensemble** | Fixed (LR+RF+SVM) | `class_weight="balanced"` |
| **DQN (RL Agent)** | 60 training episodes | PnL-shaped reward |

**Cross-validation**: `TimeSeriesSplit(n_splits=5)` (supervised)
**Scoring metric**: F1 macro (supervised) / Cumulative return (RL)
""")

    st.divider()
    st.subheader("EDA — Label Distribution & BTC Price Timeline")
    st.image(load_image("eda_label_distribution.png"), width="stretch")


def page_performance():
    st.title("Model Performance")
    st.caption("All metrics evaluated on the held-out test set (2021-01 → 2023-08)")
    st.divider()

    df = load_summary()
    best = df.loc[df["F1_macro"].idxmax()]

    st.markdown(f"**Best model by F1 (macro): &nbsp;** "
                f"{badge(best['Model'])}", unsafe_allow_html=True)
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Accuracy",   f"{df['Accuracy'].max():.4f}", f"{best['Model']}")
    c2.metric("Best F1 (macro)", f"{best['F1_macro']:.4f}",    f"{best['Model']}")
    c3.metric("Best ROC AUC",    f"{df['ROC_AUC'].max():.4f}", f"{df.loc[df['ROC_AUC'].idxmax(), 'Model']}")
    c4.metric("Models Trained",  len(df))

    st.divider()
    st.subheader("Summary Table")
    display_cols = ["Model", "Accuracy", "F1_macro", "ROC_AUC"]
    st.dataframe(
        df[display_cols]
          .rename(columns={"F1_macro": "F1 (macro)", "ROC_AUC": "ROC AUC"})
          .style
          .highlight_max(
              subset=["Accuracy", "F1 (macro)", "ROC AUC"],
              props="color: #000000; background-color: #F7931A; font-weight: 700;",
          )
          .format({"Accuracy": "{:.4f}", "F1 (macro)": "{:.4f}", "ROC AUC": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Metric Comparison")

    SHORT = {
        "Logistic Regression": "LR",
        "Random Forest": "RF",
        "XGBoost": "XGB",
        "LightGBM": "LGB",
        "SVM": "SVM",
        "KNN": "KNN",
    }
    tab1, tab2, tab3 = st.tabs(["Accuracy", "F1 (macro)", "ROC AUC"])
    for tab, col, label in zip(
        [tab1, tab2, tab3],
        ["Accuracy", "F1_macro", "ROC_AUC"],
        ["Accuracy", "F1 (macro)", "ROC AUC"],
    ):
        with tab:
            chart_df = (df.set_index("Model")[[col]]
                          .rename(index=SHORT, columns={col: label})
                          .sort_values(label, ascending=False))
            st.bar_chart(chart_df, color=ACCENT)

    st.divider()
    st.subheader("Full Results (with Best Hyperparameters)")
    st.dataframe(df, width="stretch", hide_index=True)


def page_confusion():
    st.title("Confusion Matrices")
    st.caption("Rows = true labels · Columns = predicted labels · "
               "Classes: Bear (−1) / Sideways (0) / Bull (1)")
    st.divider()

    all_cm_models = MODEL_NAMES + ["LSTM", "DQN"]
    all_cm_display = {**MODEL_DISPLAY, "LSTM": "LSTM", "DQN": "DQN (RL Agent)"}
    model_key = st.selectbox(
        "Select model", all_cm_models,
        format_func=lambda x: all_cm_display.get(x, x),
    )
    col, _ = st.columns([3, 2])
    with col:
        st.image(load_image(f"{model_key}_confusion_matrix.png"),
                 width="stretch")


def page_roc():
    st.title("ROC Curves")
    st.caption("One-vs-Rest per class · AUC scores shown in legend · "
               "All 6 models overlaid in each subplot")
    st.divider()
    st.image(load_image("roc_curves_all_models.png"), width="stretch")


def page_feature_importance():
    st.title("Feature Importance")
    st.caption("Top 20 features ranked by importance score")
    st.divider()

    model_key = st.selectbox("Select model", list(FEAT_IMP_FILES.keys()))
    st.image(load_image(FEAT_IMP_FILES[model_key]), width="stretch")


def page_rl():
    st.title("RL Agent — Deep Q-Network (DQN)")
    st.caption(
        "A DQN agent trained to **maximise portfolio return** rather than match a label. "
        "Reward = position × 30-day realised return. "
        "Actions: Bear (short) · Sideways (flat) · Bull (long)."
    )
    st.divider()

    dqn_cm   = os.path.join(OUTPUT_DIR, "DQN_confusion_matrix.png")
    dqn_rew  = os.path.join(OUTPUT_DIR, "DQN_training_rewards.png")
    dqn_cum  = os.path.join(OUTPUT_DIR, "DQN_cumulative_return.png")

    if not os.path.exists(dqn_cm):
        st.warning(
            "DQN output not found. Re-run training to generate RL results:  \n"
            "`python -m src.main`"
        )
        return

    # ── classification metrics from summary ──────────────────────────────────
    df = load_summary()
    dqn_row = df[df["Model"] == "DQN (RL)"]
    if not dqn_row.empty:
        row = dqn_row.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("DQN Accuracy", f"{row['Accuracy']:.4f}")
        c2.metric("DQN F1 (macro)", f"{row['F1_macro']:.4f}")
        c3.metric("Epochs trained", "60")
        st.divider()

    # ── architecture description ──────────────────────────────────────────────
    with st.expander("DQN Architecture & Training Details", expanded=False):
        st.markdown("""
**Network**
- Input: Top-40 on-chain features (StandardScaler-normalised)
- Layer 1: Linear(40→128) + LayerNorm + ReLU + Dropout(0.2)
- Layer 2: Linear(128→64) + LayerNorm + ReLU
- Output:  Linear(64→3) Q-values for {Bear, Sideways, Bull}

**Algorithm** — Double-DQN
- Experience replay buffer (20 000 transitions)
- Hard target-network update every 200 gradient steps
- ε-greedy exploration: ε 1.0 → 0.05 (decay ×0.997)
- Optimizer: Adam (lr=3e-4, weight_decay=1e-5)
- Loss: Smooth-L1 (Huber) | Grad-clip: 1.0

**Key difference from supervised models**
The objective is cumulative *portfolio return*, not label accuracy.
The agent can learn asymmetric risk preferences (e.g. be more
conservative during uncertain regimes) that a classifier cannot capture.
""")

    st.divider()
    tab1, tab2, tab3 = st.tabs(
        ["Confusion Matrix", "Training Reward Curve", "Cumulative Return"]
    )
    with tab1:
        col, _ = st.columns([3, 2])
        with col:
            st.image(load_image("DQN_confusion_matrix.png"), width="stretch")
    with tab2:
        st.image(load_image("DQN_training_rewards.png"), width="stretch")
        st.caption("Episode reward converging upward indicates the agent is "
                   "learning to improve its position-sizing.")
    with tab3:
        st.image(load_image("DQN_cumulative_return.png"), width="stretch")
        st.caption("Orange = DQN agent PnL · Blue dashed = buy-and-hold baseline.")


# ── router ────────────────────────────────────────────────────────────────────

def page_lstm():
    st.title("LSTM — Sequential Market State Classifier")
    st.caption(
        "Bidirectional LSTM that takes the **past 30 days** of on-chain features as input, "
        "capturing temporal dependencies that tabular models miss."
    )
    st.divider()

    if not os.path.exists(os.path.join(OUTPUT_DIR, "LSTM_confusion_matrix.png")):
        st.warning("LSTM output not found. Re-run training:  \n`python -m src.main`")
        return

    df = load_summary()
    lstm_row = df[df["Model"] == "LSTM"]
    if not lstm_row.empty:
        row = lstm_row.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",   f"{row['Accuracy']:.4f}")
        c2.metric("F1 (macro)", f"{row['F1_macro']:.4f}")
        c3.metric("ROC AUC",    f"{row['ROC_AUC']:.4f}")
        st.divider()

    with st.expander("Architecture", expanded=False):
        st.markdown("""
**Input**
Past 30 days of Top-40 on-chain features → shape `(batch, 30, 40)`

**Network**
- Bidirectional LSTM: 2 layers × 128 hidden units (→ 256 after concat)
- LayerNorm + Dropout(0.3)
- Linear(256 → 3) classification head

**Training**
- Loss: CrossEntropy with class weights (inverse frequency)
- Optimizer: Adam (lr=5e-4, weight_decay=1e-4)
- LR scheduler: ReduceLROnPlateau (×0.5 after 5 epochs no improvement)
- Early stopping: patience=10 on validation loss
- Chronological 85/15 train/val split within training data

**Key advantage over tabular models**
Each prediction uses the *sequence* of the last 30 days, not just
today's snapshot — allowing the model to detect trend acceleration,
regime transitions, and momentum shifts.
""")

    tab1, tab2 = st.tabs(["Confusion Matrix", "Training Curves"])
    with tab1:
        col, _ = st.columns([3, 2])
        with col:
            st.image(load_image("LSTM_confusion_matrix.png"), width="stretch")
    with tab2:
        st.image(load_image("LSTM_training_curves.png"), width="stretch")
        st.caption("Left: train/val loss · Right: validation F1 macro per epoch")


if   page == "Overview":           page_overview()
elif page == "Model Performance":  page_performance()
elif page == "Confusion Matrix":   page_confusion()
elif page == "ROC Curves":         page_roc()
elif page == "Feature Importance": page_feature_importance()
elif page == "LSTM":               page_lstm()
elif page == "RL Agent (DQN)":     page_rl()
