# pages/02_Plot_Correlations.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from pathlib import Path

st.set_page_config(page_title="Plot Correlations", layout="wide")
st.title("Plot Correlations")

def _load_df() -> pd.DataFrame | None:
    if "raw_df" in st.session_state and isinstance(st.session_state["raw_df"], pd.DataFrame):
        return st.session_state["raw_df"]
    default_path = "quadrant_analysis_data.csv"
    if Path(default_path).exists():
        st.info("Loaded dataset from default path (new tab / new session).")
        return pd.read_csv(default_path)
    st.warning("No dataset found. Upload the CSV here or open the main page first.")
    up = st.file_uploader("Upload CSV", type=["csv"], key="corr_upload")
    if up is not None:
        return pd.read_csv(up)
    return None

df = _load_df()
if df is None:
    st.stop()

# numeric-only candidates
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
if not num_cols:
    st.error("No numeric columns found to compute correlations.")
    st.stop()

# --------------------------
# UI — axis builders
# --------------------------
def build_axis_series(prefix: str) -> pd.Series | None:
    st.subheader(f"{prefix.upper()} axis")

    cols_sel = st.multiselect(
        f"Select {prefix}-axis column(s)",
        options=num_cols,
        key=f"{prefix}_cols",
    )
    if not cols_sel:
        st.info(f"Pick at least one numeric column for the {prefix}-axis.")
        return None

    agg = st.selectbox(
        f"Aggregation for {prefix}-axis (when multiple columns selected)",
        options=["Sum", "Mean", "Weighted Sum"],
        index=0,
        key=f"{prefix}_agg",
    )

    if len(cols_sel) == 1:
        # single column: just return it (ignores agg choice)
        s = df[cols_sel[0]].copy()
        s.name = cols_sel[0]
        return s

    if agg == "Sum":
        s = df[cols_sel].sum(axis=1, min_count=1)
    elif agg == "Mean":
        s = df[cols_sel].mean(axis=1)
    else:
        st.caption("Weights will be normalized so their sum equals 1.")
        w = {}
        cols = st.columns(min(4, max(1, len(cols_sel))))
        for i, c in enumerate(cols_sel):
            with cols[i % len(cols)]:
                w[c] = st.number_input(
                    f"Weight · {c}",
                    min_value=0.0, value=1.0, step=0.1,
                    key=f"{prefix}_w_{c}",
                    format="%.6f",
                )
        wsum = sum(w.values())
        if wsum <= 0:
            st.warning("Weights sum to 0; using equal weights.")
            w = {c: 1.0 for c in cols_sel}
            wsum = float(len(cols_sel))
        wnorm = {c: v / wsum for c, v in w.items()}
        parts = []
        for c in cols_sel:
            parts.append(wnorm[c] * df[c])
        s = sum(parts)

    s.name = f"{' + '.join(cols_sel)} ({agg})"
    return s

# --------------------------
# Build X and Y, correlation & plot
# --------------------------
left, right = st.columns(2)
with left:
    sx = build_axis_series("x")
with right:
    sy = build_axis_series("y")

if (sx is None) or (sy is None):
    st.stop()

# align/dropna pairwise
pair = pd.concat([sx, sy], axis=1).dropna()
pair.columns = ["X", "Y"]

if pair.empty:
    st.error("No overlapping non-NaN rows for the selected columns.")
    st.stop()

method = st.selectbox(
    "Correlation method",
    # options=["Pearson", "Spearman", "Kendall"],
    options=["Pearson"],
    index=0,
    key="corr_method",
)

r = pair["X"].corr(pair["Y"], method=method.lower())

def corr_strength(r: float) -> tuple[str, str]:
    a = abs(r)
    if a < 0.10:   return ("Negligible", "#94a3b8")  # slate
    if a < 0.30:   return ("Weak", "#f59e0b")        # amber
    if a < 0.50:   return ("Moderate", "#3b82f6")    # blue
    if a < 0.70:   return ("Strong", "#22c55e")      # green
    return ("Very strong", "#9333ea")                # purple

kind = "Positive" if r > 0.05 else ("Negative" if r < -0.05 else "None")
strength, color = corr_strength(r)

# Info chips
st.markdown(
    f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:6px">
  <div style="padding:6px 10px;border:1px solid #d0d4da;border-radius:999px">
    <b>r</b> = {r:.4f} ({method})
  </div>
  <div style="padding:6px 10px;border:1px solid #d0d4da;border-radius:999px">
    Type: <b>{kind}</b>
  </div>
  <div style="padding:6px 10px;border:1px solid #d0d4da;border-radius:999px;background:{color}20">
    Strength: <b style="color:{color}">{strength}</b>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Scatter + best-fit line (least squares on X,Y)
x = pair["X"].values
y = pair["Y"].values

# fit line regardless of method; it's a visual guide
try:
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    show_line = True
except Exception:
    show_line = False

fig = go.Figure()
fig.add_trace(
    go.Scattergl(
        x=x, y=y,
        mode="markers",
        marker=dict(size=7),
        name="Points",
        hovertemplate="X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>",
    )
)
if show_line:
    fig.add_trace(
        go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            line=dict(width=2),
            name="Best-fit line",
            hoverinfo="skip",
        )
    )

fig.update_layout(
    xaxis_title=sx.name or "X",
    yaxis_title=sy.name or "Y",
    height=650,
    margin=dict(l=40, r=20, t=30, b=40),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

st.plotly_chart(fig, use_container_width=True)
