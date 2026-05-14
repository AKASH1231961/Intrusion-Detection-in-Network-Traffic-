import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COLNAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "last_flag",
]

# 23 features selected by mutual information (MI > 0.1) during training
FEATURE_COLS = [
    "src_bytes", "diff_srv_rate", "same_srv_rate", "dst_bytes",
    "dst_host_diff_srv_rate", "dst_host_srv_count", "count",
    "dst_host_serror_rate", "dst_host_same_srv_rate", "serror_rate",
    "flag_SF", "dst_host_srv_serror_rate", "srv_serror_rate", "flag_S0",
    "logged_in", "dst_host_srv_diff_host_rate", "dst_host_same_src_port_rate",
    "dst_host_count", "service_http", "srv_count", "srv_diff_host_rate",
    "service_private", "dst_host_rerror_rate",
]

# Subset of FEATURE_COLS that are numerical (the rest are OHE binary flags)
OHE_FEATURES = {"flag_SF", "flag_S0", "service_http", "service_private"}
NUM_FEATURES = [f for f in FEATURE_COLS if f not in OHE_FEATURES]

# LabelEncoder — must exactly match notebook's fit order
ALL_CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r", "other"]
_le = LabelEncoder()
_le.fit(ALL_CATEGORIES)
LABEL_MAP = {int(code): cls for code, cls in zip(_le.transform(_le.classes_), _le.classes_)}

LABEL_COLORS = {
    "normal": "#22c55e",
    "dos":    "#ef4444",
    "probe":  "#f59e0b",
    "r2l":    "#8b5cf6",
    "u2r":    "#ec4899",
    "other":  "#6b7280",
}

LABEL_DESCRIPTIONS = {
    "normal": "Normal traffic — no intrusion detected.",
    "dos":    "Denial of Service attack — resource exhaustion attempt.",
    "probe":  "Probe / Surveillance — scanning for vulnerabilities.",
    "r2l":    "Remote-to-Local — unauthorised remote access attempt.",
    "u2r":    "User-to-Root — privilege escalation attempt.",
    "other":  "Unclassified attack type.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Resource loading (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models & fitting scaler…")
def load_resources():
    with open("random_forest_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("xgboost_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open("mlp_model.pkl", "rb") as f:
        mlp = pickle.load(f)

    # StandardScaler was NOT saved in the repo; refit it from training data.
    # StandardScaler is column-independent (mean/std per column), so fitting
    # on just the 19 numerical features gives identical parameters to the
    # original fit on the full numerical set.
    train_df = pd.read_csv("KDDTrain+.txt", names=COLNAMES)
    scaler = StandardScaler()
    scaler.fit(train_df[NUM_FEATURES])

    return {"rf": rf, "xgb": xgb, "mlp": mlp, "scaler": scaler}


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_row(values: dict) -> dict:
    """
    Turn a dict of raw KDD field values into a dict keyed by FEATURE_COLS.
    Handles OHE for 'flag' and 'service'.
    """
    flag    = str(values.get("flag",    "OTHER"))
    service = str(values.get("service", "other"))

    row = {}
    for col in FEATURE_COLS:
        if col in OHE_FEATURES:
            row[col] = 0.0
        else:
            row[col] = float(values.get(col, 0.0))

    row["flag_SF"]       = 1.0 if flag    == "SF"      else 0.0
    row["flag_S0"]       = 1.0 if flag    == "S0"      else 0.0
    row["service_http"]  = 1.0 if service == "http"    else 0.0
    row["service_private"] = 1.0 if service == "private" else 0.0

    return row


def preprocess(rows: list[dict], scaler: StandardScaler) -> pd.DataFrame:
    """Scale numerical features and return a DataFrame ready for prediction."""
    df = pd.DataFrame([build_feature_row(r) for r in rows])[FEATURE_COLS]
    df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])
    return df


def predict_single(df_row: pd.DataFrame, model) -> tuple[str, dict]:
    proba   = model.predict_proba(df_row)[0]
    classes = model.classes_
    idx     = int(np.argmax(proba))
    label   = LABEL_MAP.get(int(classes[idx]), "unknown")
    proba_dict = {LABEL_MAP.get(int(c), str(c)): float(p)
                  for c, p in zip(classes, proba)}
    return label, proba_dict


def preprocess_raw_csv(df_raw: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Preprocess a raw KDD-format CSV (all original columns, no labels)."""
    rows = df_raw.to_dict(orient="records")
    return preprocess(rows, scaler)


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def result_badge(label: str) -> str:
    color = LABEL_COLORS.get(label, "#6b7280")
    return (
        f'<span style="background:{color};color:white;padding:6px 18px;'
        f'border-radius:999px;font-weight:700;font-size:1.1rem;">'
        f'{label.upper()}</span>'
    )


def probability_chart(proba_dict: dict) -> go.Figure:
    labels = list(proba_dict.keys())
    values = [proba_dict[l] * 100 for l in labels]
    colors = [LABEL_COLORS.get(l, "#6b7280") for l in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(title="Confidence (%)", range=[0, 110]),
        xaxis_title="Attack Category",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20),
        height=300,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NIDS — Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #1e293b; border-radius: 12px; padding: 1.2rem 1.5rem;
        border-left: 4px solid #3b82f6;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ Network Intrusion Detection System")
st.markdown(
    "Classify network connections as **normal** or one of four attack categories "
    "(*DoS · Probe · R2L · U2R*) using Random Forest, XGBoost, or MLP."
)
st.divider()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.radio(
        "Model",
        ["Random Forest", "XGBoost", "MLP"],
        index=0,
        help="All three models were trained on NSL-KDD with mutual-information feature selection.",
    )
    st.divider()
    st.markdown("**Dataset:** NSL-KDD  \n**Features used:** 23 (MI > 0.1)  \n**Classes:** Normal · DoS · Probe · R2L · U2R")
    st.divider()
    st.markdown("**Attack categories**")
    for label, desc in LABEL_DESCRIPTIONS.items():
        color = LABEL_COLORS[label]
        st.markdown(
            f'<span style="color:{color};font-weight:700">{label.upper()}</span> — {desc}',
            unsafe_allow_html=True,
        )

# ── Load resources ───────────────────────────────────────────────────────────
res = load_resources()
model_map = {"Random Forest": res["rf"], "XGBoost": res["xgb"], "MLP": res["mlp"]}
model = model_map[model_choice]
scaler = res["scaler"]

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_manual, tab_batch = st.tabs(["🔍 Single Connection", "📂 Batch CSV Upload"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Manual single-connection input
# ════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("Inspect a single network connection")
    st.markdown(
        "Fill in the fields below. Only the **23 features** selected during training "
        "are shown. Flags and services not listed default to *other*."
    )

    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Traffic volume**")
            src_bytes = st.number_input("src_bytes",  min_value=0, value=0,
                                        help="Bytes sent from source")
            dst_bytes = st.number_input("dst_bytes",  min_value=0, value=0,
                                        help="Bytes sent from destination")
            count     = st.number_input("count",      min_value=0, max_value=511, value=1,
                                        help="Connections to same host in last 2 sec")
            srv_count = st.number_input("srv_count",  min_value=0, max_value=511, value=1,
                                        help="Connections to same service in last 2 sec")
            dst_host_count     = st.number_input("dst_host_count",     min_value=0, max_value=255, value=1)
            dst_host_srv_count = st.number_input("dst_host_srv_count", min_value=0, max_value=255, value=1)

        with c2:
            st.markdown("**Error & service rates**")
            serror_rate             = st.slider("serror_rate",             0.0, 1.0, 0.0, 0.01)
            srv_serror_rate         = st.slider("srv_serror_rate",         0.0, 1.0, 0.0, 0.01)
            rerror_rate_val         = st.slider("rerror_rate (dst_host)",  0.0, 1.0, 0.0, 0.01)
            dst_host_serror_rate    = st.slider("dst_host_serror_rate",    0.0, 1.0, 0.0, 0.01)
            dst_host_srv_serror_rate= st.slider("dst_host_srv_serror_rate",0.0, 1.0, 0.0, 0.01)
            same_srv_rate           = st.slider("same_srv_rate",           0.0, 1.0, 1.0, 0.01)
            diff_srv_rate           = st.slider("diff_srv_rate",           0.0, 1.0, 0.0, 0.01)

        with c3:
            st.markdown("**Destination host rates**")
            dst_host_same_srv_rate      = st.slider("dst_host_same_srv_rate",      0.0, 1.0, 1.0, 0.01)
            dst_host_diff_srv_rate      = st.slider("dst_host_diff_srv_rate",      0.0, 1.0, 0.0, 0.01)
            dst_host_same_src_port_rate = st.slider("dst_host_same_src_port_rate", 0.0, 1.0, 0.0, 0.01)
            dst_host_srv_diff_host_rate = st.slider("dst_host_srv_diff_host_rate", 0.0, 1.0, 0.0, 0.01)
            dst_host_rerror_rate        = st.slider("dst_host_rerror_rate",        0.0, 1.0, 0.0, 0.01)
            srv_diff_host_rate          = st.slider("srv_diff_host_rate",          0.0, 1.0, 0.0, 0.01)

        st.markdown("---")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            flag    = st.selectbox("flag",    ["SF", "S0", "REJ", "RSTO", "SH", "RSTR",
                                               "S1", "S2", "S3", "OTH"],
                                   help="TCP flag — SF=normal, S0=no response")
        with cc2:
            service = st.selectbox("service", ["http", "private", "ftp_data", "smtp",
                                               "ftp", "ssh", "domain_u", "ecr_i", "other"],
                                   help="Network service")
        with cc3:
            logged_in = st.selectbox("logged_in", [0, 1],
                                     help="1 = user successfully logged in")

        submitted = st.form_submit_button("🔎 Classify", use_container_width=True, type="primary")

    if submitted:
        row = {
            "src_bytes": src_bytes, "dst_bytes": dst_bytes,
            "count": count, "srv_count": srv_count,
            "dst_host_count": dst_host_count, "dst_host_srv_count": dst_host_srv_count,
            "serror_rate": serror_rate, "srv_serror_rate": srv_serror_rate,
            "dst_host_rerror_rate": rerror_rate_val,
            "dst_host_serror_rate": dst_host_serror_rate,
            "dst_host_srv_serror_rate": dst_host_srv_serror_rate,
            "same_srv_rate": same_srv_rate, "diff_srv_rate": diff_srv_rate,
            "dst_host_same_srv_rate": dst_host_same_srv_rate,
            "dst_host_diff_srv_rate": dst_host_diff_srv_rate,
            "dst_host_same_src_port_rate": dst_host_same_src_port_rate,
            "dst_host_srv_diff_host_rate": dst_host_srv_diff_host_rate,
            "srv_diff_host_rate": srv_diff_host_rate,
            "logged_in": logged_in,
            "flag": flag, "service": service,
        }
        df_input = preprocess([row], scaler)
        label, proba = predict_single(df_input, model)

        st.markdown("---")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown("#### Prediction")
            st.markdown(result_badge(label), unsafe_allow_html=True)
            st.markdown(f"**Confidence:** `{proba[label]*100:.1f}%`")
            st.markdown(f"_{LABEL_DESCRIPTIONS.get(label, '')}_")
        with r2:
            st.markdown("#### Probability distribution")
            st.plotly_chart(probability_chart(proba), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch CSV upload
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Classify a batch of connections from a CSV file")

    with st.expander("📋 Expected CSV format", expanded=False):
        st.markdown(
            "Upload a headerless CSV in **KDD format** (41 or 43 columns).  \n"
            "The first 41 columns match the original KDD feature set:\n\n"
            "`duration, protocol_type, service, flag, src_bytes, dst_bytes, ...`\n\n"
            "Attack label columns (`attack_type`, `last_flag`) are optional and will be ignored."
        )
        st.code(", ".join(COLNAMES[:43]))

    uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])

    if uploaded:
        try:
            raw = pd.read_csv(uploaded, header=None)
            # Accept 41-col (no labels) or 43-col (with labels)
            if raw.shape[1] == 43:
                raw.columns = COLNAMES
                raw = raw.drop(columns=["attack_type", "last_flag"], errors="ignore")
            elif raw.shape[1] == 41:
                raw.columns = COLNAMES[:41]
            else:
                st.error(f"Expected 41 or 43 columns, got {raw.shape[1]}.")
                st.stop()

            st.success(f"Loaded **{len(raw):,}** connections.")

            with st.spinner("Running inference…"):
                df_proc = preprocess_raw_csv(raw, scaler)
                preds   = model.predict(df_proc)
                probas  = model.predict_proba(df_proc)

            labels = [LABEL_MAP.get(int(p), "unknown") for p in preds]
            conf   = [float(probas[i, np.argmax(probas[i])]) for i in range(len(preds))]

            results_df = raw.copy()
            results_df["predicted_category"] = labels
            results_df["confidence"]         = [f"{c*100:.1f}%" for c in conf]

            # ── Summary metrics ─────────────────────────────────────────────
            st.markdown("### Results summary")
            counts = pd.Series(labels).value_counts()
            cols   = st.columns(min(len(counts), 6))
            for i, (lbl, cnt) in enumerate(counts.items()):
                with cols[i % len(cols)]:
                    color = LABEL_COLORS.get(lbl, "#6b7280")
                    pct   = cnt / len(labels) * 100
                    st.markdown(
                        f'<div class="metric-card" style="border-color:{color}">'
                        f'<div style="font-size:1.6rem;font-weight:700;color:{color}">{cnt}</div>'
                        f'<div style="font-size:0.85rem;color:#94a3b8">{lbl.upper()} ({pct:.1f}%)</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # ── Pie chart ────────────────────────────────────────────────────
            st.markdown("### Distribution")
            fig_pie = px.pie(
                values=counts.values,
                names=counts.index,
                color=counts.index,
                color_discrete_map=LABEL_COLORS,
                hole=0.45,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=20), height=340,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # ── Full table ───────────────────────────────────────────────────
            with st.expander("📄 Full results table", expanded=False):
                st.dataframe(
                    results_df[["src_bytes", "dst_bytes", "flag", "service",
                                "predicted_category", "confidence"]],
                    use_container_width=True,
                )

            # ── Download ─────────────────────────────────────────────────────
            csv_out = results_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download full results as CSV",
                data=csv_out,
                file_name="nids_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
