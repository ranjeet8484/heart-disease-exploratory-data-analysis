import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Heart Disease EDA Report",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# Page styling
# ----------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 1rem;
        }

        .hero {
            padding: 2rem 2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            border: 1px solid #dbe4f0;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
            margin-bottom: 16px;
        }

        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            color: #0f172a;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: #475569;
            line-height: 1.7;
            max-width: 980px;
        }

        .section-label {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.03em;
            color: #334155;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .insight-box {
            padding: 1rem 1rem;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            background: #f8fafc;
            color: #0f172a !important;
            border-left: 5px solid #2563eb;
            line-height: 1.6;
            margin-bottom: 0.8rem;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
        }

        .insight-box strong,
        .insight-box span,
        .insight-box div {
            color: #0f172a !important;
        }

        .muted {
            color: #64748b;
            font-size: 0.92rem;
        }

        .small-note {
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        /* Make default Streamlit elements a touch cleaner */
        div[data-testid="stMetric"] {
            background: rgba(248, 250, 252, 0.70);
            border: 1px solid #e2e8f0;
            padding: 0.9rem 1rem;
            border-radius: 16px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricLabel"] > div {
            color: #334155;
            font-size: 0.9rem;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 800;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Data utilities
# ----------------------------
DATA_PATH = Path(__file__).parent / "heart.csv"

SEX_MAP = {0: "Female", 1: "Male"}
CP_MAP = {
    0: "Typical angina",
    1: "Atypical angina",
    2: "Non-anginal pain",
    3: "Asymptomatic",
}
FBS_MAP = {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"}
RESTECG_MAP = {
    0: "Normal",
    1: "ST-T abnormality",
    2: "Left ventricular hypertrophy",
}
EXANG_MAP = {0: "No", 1: "Yes"}
SLOPE_MAP = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
THAL_MAP = {
    0: "Unknown",
    1: "Normal",
    2: "Fixed defect",
    3: "Reversible defect",
}
TARGET_MAP = {0: "No disease", 1: "Disease"}

FEATURE_GUIDE = pd.DataFrame(
    [
        ["age", "29–77 (years)", "Numerical – Continuous (Ratio)", "Scale (optional)", "Age of the patient in years"],
        ["sex", "0 = Female, 1 = Male", "Categorical – Binary (Nominal)", "Keep as-is", "Biological sex of the patient"],
        ["cp", "0–3 chest pain categories", "Categorical – Nominal", "One-hot encode for non-tree models", "Type of chest pain experienced"],
        ["trestbps", "94–200 mm Hg", "Numerical – Continuous (Ratio)", "Scale (optional)", "Resting blood pressure"],
        ["chol", "126–564 mg/dl", "Numerical – Continuous (Ratio)", "Scale / optional log", "Serum cholesterol level"],
        ["fbs", "0 = ≤120 mg/dl, 1 = >120 mg/dl", "Categorical – Binary (Nominal)", "Keep as-is", "Fasting blood sugar indicator"],
        ["restecg", "0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy", "Categorical – Nominal", "One-hot encode", "Resting electrocardiographic result"],
        ["thalach", "71–202 bpm", "Numerical – Continuous (Ratio)", "Scale", "Maximum heart rate achieved"],
        ["exang", "0 = No, 1 = Yes", "Categorical – Binary (Nominal)", "Keep or drop (diagnostic risk)", "Exercise-induced angina"],
        ["oldpeak", "0.0–6.2", "Numerical – Continuous (Ratio)", "Scale", "ST depression induced by exercise"],
        ["slope", "0 = Upsloping, 1 = Flat, 2 = Downsloping", "Categorical – Ordinal", "Keep ordinal or one-hot", "Slope of peak exercise ST segment"],
        ["ca", "0–3 vessels", "Numerical – Discrete (Ordinal)", "Drop for baseline / keep for diagnostic", "Number of major vessels colored by fluoroscopy"],
        ["thal", "1 = Normal, 2 = Fixed defect, 3 = Reversible defect", "Categorical – Nominal", "One-hot or drop (leakage risk)", "Thalassemia blood disorder status"],
        ["target", "0 = No disease, 1 = Disease", "Categorical – Binary (Nominal)", "Target variable", "Presence of heart disease"],
    ],
    columns=["Name", "Values", "Type", "Action (ML Processing)", "Feature Meaning"],
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.name} beside app.py.")
    return pd.read_csv(DATA_PATH)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sex_label"] = out["sex"].map(SEX_MAP).fillna(out["sex"].astype(str))
    out["cp_label"] = out["cp"].map(CP_MAP).fillna(out["cp"].astype(str))
    out["fbs_label"] = out["fbs"].map(FBS_MAP).fillna(out["fbs"].astype(str))
    out["restecg_label"] = out["restecg"].map(RESTECG_MAP).fillna(out["restecg"].astype(str))
    out["exang_label"] = out["exang"].map(EXANG_MAP).fillna(out["exang"].astype(str))
    out["slope_label"] = out["slope"].map(SLOPE_MAP).fillna(out["slope"].astype(str))
    out["thal_label"] = out["thal"].map(THAL_MAP).fillna(out["thal"].astype(str))
    out["target_label"] = out["target"].map(TARGET_MAP).fillna(out["target"].astype(str))

    bins = [28, 39, 49, 59, 69, 120]
    labels = ["29–39", "40–49", "50–59", "60–69", "70+"]
    out["age_group"] = pd.cut(out["age"], bins=bins, labels=labels, include_lowest=True, right=True)
    out["age_group"] = out["age_group"].astype(str).replace("nan", "Unknown")
    return out


def disease_rate(series: pd.Series) -> float:
    return float(series.mean() * 100) if len(series) else 0.0


def nice_bar(fig):
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=40, r=35, t=60, b=40),
        legend_title_text="",
        font=dict(size=14),
        title_font_size=20,
    )
    return fig


def build_insights(df: pd.DataFrame) -> list[str]:
    insights = []

    if len(df) == 0:
        return ["No records match the current filters."]

    target_rate = disease_rate(df["target"])
    insights.append(f"Disease prevalence in the current view is {target_rate:.1f}%.")

    by_sex = df.groupby("sex_label")["target"].mean().sort_values(ascending=False) * 100
    if len(by_sex) >= 2:
        top_sex = by_sex.index[0]
        bottom_sex = by_sex.index[-1]
        gap = by_sex.iloc[0] - by_sex.iloc[-1]
        insights.append(
            f"{top_sex} shows a higher disease rate than {bottom_sex} by {gap:.1f} percentage points."
        )

    by_age = df.groupby("age_group")["target"].mean().sort_values(ascending=False) * 100
    if len(by_age):
        insights.append(
            f"Highest disease rate appears in the {by_age.index[0]} age group ({by_age.iloc[0]:.1f}%)."
        )

    by_cp = df.groupby("cp_label")["target"].mean().sort_values(ascending=False) * 100
    if len(by_cp):
        insights.append(
            f"Chest pain type most associated with disease here: {by_cp.index[0]} ({by_cp.iloc[0]:.1f}%)."
        )

    return insights[:4]


def data_for_download(df: pd.DataFrame) -> bytes:
    export_df = df.drop(columns=[c for c in df.columns if c.endswith("_label") or c == "age_group"])
    return export_df.to_csv(index=False).encode("utf-8")


# ----------------------------
# Load and prepare
# ----------------------------
raw = load_data()
df = add_labels(raw)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.markdown("## Filters")
st.sidebar.caption("Use the filters to explore the EDA interactively.")

sex_options = sorted(df["sex_label"].dropna().unique().tolist())
cp_options = sorted(df["cp_label"].dropna().unique().tolist())
target_options = sorted(df["target_label"].dropna().unique().tolist())
exang_options = sorted(df["exang_label"].dropna().unique().tolist())
thal_options = sorted(df["thal_label"].dropna().unique().tolist())
restecg_options = sorted(df["restecg_label"].dropna().unique().tolist())
slope_options = sorted(df["slope_label"].dropna().unique().tolist())

age_min, age_max = int(df["age"].min()), int(df["age"].max())
chol_min, chol_max = int(df["chol"].min()), int(df["chol"].max())
thalach_min, thalach_max = int(df["thalach"].min()), int(df["thalach"].max())
bp_min, bp_max = int(df["trestbps"].min()), int(df["trestbps"].max())

selected_age = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
selected_sex = st.sidebar.multiselect("Sex", sex_options, default=sex_options)
selected_target = st.sidebar.multiselect("Target", target_options, default=target_options)
selected_cp = st.sidebar.multiselect("Chest pain type", cp_options, default=cp_options)
selected_exang = st.sidebar.multiselect("Exercise angina", exang_options, default=exang_options)
selected_thal = st.sidebar.multiselect("Thal", thal_options, default=thal_options)
selected_restecg = st.sidebar.multiselect("Restecg", restecg_options, default=restecg_options)
selected_slope = st.sidebar.multiselect("Slope", slope_options, default=slope_options)

selected_chol = st.sidebar.slider("Cholesterol", chol_min, chol_max, (chol_min, chol_max))
selected_thalach = st.sidebar.slider("Max heart rate", thalach_min, thalach_max, (thalach_min, thalach_max))
selected_bp = st.sidebar.slider("Resting BP", bp_min, bp_max, (bp_min, bp_max))

filtered = df[
    df["age"].between(*selected_age)
    & df["sex_label"].isin(selected_sex)
    & df["target_label"].isin(selected_target)
    & df["cp_label"].isin(selected_cp)
    & df["exang_label"].isin(selected_exang)
    & df["thal_label"].isin(selected_thal)
    & df["restecg_label"].isin(selected_restecg)
    & df["slope_label"].isin(selected_slope)
    & df["chol"].between(*selected_chol)
    & df["thalach"].between(*selected_thalach)
    & df["trestbps"].between(*selected_bp)
].copy()
full_age_view = selected_age == (age_min, age_max)

# ----------------------------
# Header / hero
# ----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🫀 Heart Disease Exploratory Data Analysis</div>
        <div class="hero-subtitle">
            An interactive Streamlit report that turns the exploratory analysis into a recruiter-friendly story:
            patient demographics, clinical indicators, correlations, and feature-level insights — without building a predictive model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ----------------------------
# Top KPIs
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Records", f"{len(filtered):,}")
with k2:
    st.metric("Disease Rate", f"{filtered['target'].mean() * 100:.1f}%" if len(filtered) else "0.0%")
with k3:
    st.metric("Avg Age", f"{filtered['age'].mean():.1f}" if len(filtered) else "—")
with k4:
    st.metric("Avg Max HR", f"{filtered['thalach'].mean():.1f}" if len(filtered) else "—")

k5, k6 = st.columns(2)
with k5:
    st.metric("Avg Cholesterol", f"{filtered['chol'].mean():.1f}" if len(filtered) else "—")
with k6:
    st.metric("Avg Resting BP", f"{filtered['trestbps'].mean():.1f}" if len(filtered) else "—")

st.caption(
    "Dataset: 1,025 patient records • 14 columns • cleaned and explored through statistical summaries, correlations, and visual analysis."
)

# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_clinical, tab_corr, tab_guide, tab_data = st.tabs(
    ["Overview", "Clinical Patterns", "Correlations", "Feature Guide", "Data Table"]
)

# ----------------------------
# Overview tab
# ----------------------------
with tab_overview:
    c1, c2 = st.columns([1.1, 1.1], gap="large")

    with c1:
        target_dist = filtered["target_label"].value_counts().reindex(["No disease", "Disease"]).fillna(0)
        fig = px.pie(
            names=target_dist.index,
            values=target_dist.values,
            hole=0.6,
            title="Target distribution",
            color_discrete_sequence=["#64748b", "#2563eb"],
        )
        fig.update_traces(textinfo="percent+label")
        nice_bar(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_sex = (
            filtered.groupby("sex_label")["target"]
            .mean()
            .reindex(["Female", "Male"])
            .mul(100)
            .reset_index(name="Disease rate (%)")
            .dropna()
        )
        fig = px.bar(
            by_sex,
            x="sex_label",
            y="Disease rate (%)",
            title="Disease rate by sex",
            text_auto=".1f",
            color="sex_label",
            color_discrete_sequence=["#0f766e", "#7c3aed"],
        )
        fig.update_layout(showlegend=False)
        nice_bar(fig)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns([1.1, 1.1], gap="large")

    with c3:

        by_age = (
            filtered.groupby("age_group")["target"]
            .mean()
            .mul(100)
            .reset_index(name="Disease rate (%)")
            .dropna()
        )

        age_order = ["29–39", "40–49", "50–59", "60–69", "70+"]
        by_age["age_group"] = pd.Categorical(
            by_age["age_group"],
            categories=age_order,
            ordered=True,
        )
        by_age = by_age.sort_values("age_group")

    # Dynamic title
        if selected_age == (age_min, age_max):
            chart_title = "Disease rate by age group"
        else:
            chart_title = (
                f"Disease rate across predefined age groups "
                f"(Selected ages: {selected_age[0]}–{selected_age[1]})"
        )

        fig = px.bar(
            by_age,
            x="age_group",
            y="Disease rate (%)",
            title=chart_title,
            text_auto=".1f",
            color_discrete_sequence=["#2563eb"],
    )
    
        fig.update_layout(
            xaxis_title="Age Group",
            yaxis_title="Disease rate (%)",
            margin=dict(l=40, r=40, t=70, b=40),
        )

        nice_bar(fig)
        st.plotly_chart(fig, use_container_width=True)

        if selected_age != (age_min, age_max):
            st.caption(
                "Note: Bars represent the predefined age bands that overlap with the selected age range."
            )
    with c4:
        insights = build_insights(filtered)
        st.markdown('<div class="section-label">Key insights</div>', unsafe_allow_html=True)
        for idx, text in enumerate(insights, start=1):
            st.markdown(
                f"""
                <div class="insight-box">
                    <div style="color:#0f172a !important;">
                        <strong>{idx}.</strong> {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        

# ----------------------------
# Clinical patterns tab
# ----------------------------
with tab_clinical:
    st.markdown('<div class="section-label">Clinical feature comparison</div>', unsafe_allow_html=True)
    feature_choice = st.selectbox(
        "Choose a categorical feature to compare disease prevalence",
        ["cp_label", "restecg_label", "slope_label", "thal_label", "exang_label", "fbs_label"],
        format_func=lambda x: {
            "cp_label": "Chest pain type",
            "restecg_label": "Resting ECG result",
            "slope_label": "Slope",
            "thal_label": "Thal",
            "exang_label": "Exercise-induced angina",
            "fbs_label": "Fasting blood sugar",
        }[x],
    )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        cat_map = (
            filtered.groupby(feature_choice)["target"]
            .mean()
            .mul(100)
            .reset_index(name="Disease rate (%)")
            .sort_values("Disease rate (%)", ascending=False)
        )
        fig = px.bar(
            cat_map,
            x=feature_choice,
            y="Disease rate (%)",
            title="Disease rate across categories",
            text_auto=".1f",
            color="Disease rate (%)",
            color_continuous_scale=["#ede9fe", "#7c3aed"],
        )
        fig.update_layout(xaxis_title="", yaxis_title="Disease rate (%)")
        nice_bar(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        num_choice = st.selectbox(
            "Choose a numeric feature",
            ["thalach", "oldpeak", "chol", "trestbps"],
            format_func=lambda x: {
                "thalach": "Maximum heart rate (thalach)",
                "oldpeak": "ST depression (oldpeak)",
                "chol": "Serum cholesterol (chol)",
                "trestbps": "Resting blood pressure (trestbps)",
            }[x],
        )
        fig = px.box(
            filtered,
            x="target_label",
            y=num_choice,
            color="target_label",
            title=f"{num_choice} by target",
            color_discrete_sequence=["#64748b", "#2563eb"],
            points="outliers",
        )
        fig.update_layout(xaxis_title="", yaxis_title=num_choice)
        nice_bar(fig)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2, gap="large")
    with col_c:
        fig = px.scatter(
            filtered,
            x="thalach",
            y="oldpeak",
            color="target_label",
            size="age",
            hover_data=["age", "sex_label", "cp_label", "exang_label"],
            title="Maximum heart rate vs ST depression",
            color_discrete_sequence=["#64748b", "#2563eb"],
        )
        fig.update_layout(xaxis_title="Maximum heart rate", yaxis_title="ST depression (oldpeak)")
        nice_bar(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        cp = (
            filtered.groupby("cp_label")["target"]
            .mean()
            .mul(100)
            .reset_index(name="Disease rate (%)")
            .sort_values("Disease rate (%)", ascending=False)
        )
        fig = px.bar(
            cp,
            x="cp_label",
            y="Disease rate (%)",
            title="Chest pain type and disease rate",
            text_auto=".1f",
            color="Disease rate (%)",
            color_continuous_scale=["#dbeafe", "#1d4ed8"],
        )
        fig.update_layout(xaxis_title="", yaxis_title="Disease rate (%)")
        nice_bar(fig)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Correlations tab
# ----------------------------
with tab_corr:
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "target"]
    corr = filtered[numeric_cols].corr(numeric_only=True)

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="Blues",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Corr"),
            )
        )
        fig.update_layout(title="Correlation heatmap")
        nice_bar(fig)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        target_corr = (
            corr["target"]
            .drop("target")
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .reset_index()
        )
        target_corr.columns = ["Feature", "Correlation with target"]
        fig = px.bar(
            target_corr,
            x="Correlation with target",
            y="Feature",
            orientation="h",
            title="Features most associated with target",
            text_auto=".2f",
            color="Correlation with target",
            color_continuous_scale=["#fee2e2", "#2563eb"],
        )
        nice_bar(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Short interpretation")
    st.write(
        "The report emphasizes that maximum heart rate, exercise-induced angina, chest pain type, and ST depression carry useful signal, "
        "while several variables show weaker individual correlations and should be interpreted together rather than alone."
    )

# ----------------------------
# Feature guide tab
# ----------------------------
with tab_guide:
    st.markdown(
        "This table mirrors the feature-level summary used in the exploratory analysis report and helps recruiters quickly understand the dataset."
    )
    st.dataframe(FEATURE_GUIDE, use_container_width=True, hide_index=True)

# ----------------------------
# Data tab
# ----------------------------
with tab_data:
    st.markdown("### Filtered data preview")
    st.write(f"Showing **{len(filtered):,}** rows after applying the sidebar filters.")
    st.dataframe(
        filtered.drop(columns=[c for c in filtered.columns if c.endswith("_label") or c == "age_group"]),
        use_container_width=True,
        height=420,
    )

    st.download_button(
        "Download filtered CSV",
        data=data_for_download(filtered),
        file_name="heart_disease_filtered.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#64748b;padding:14px;font-size:15px;">
    <b>Developed by Ranjeet Paswan</b> • Heart Disease Exploratory Data Analysis • Python • Streamlit • Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
