# pages/03_Correlation_Matrix.py
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Correlation Matrix", layout="wide")
st.title("Correlation Matrix")

def _load_df() -> pd.DataFrame | None:
    if "raw_df" in st.session_state and isinstance(st.session_state["raw_df"], pd.DataFrame):
        return st.session_state["raw_df"]
    default_path = "quadrant_analysis_data.csv"
    if Path(default_path).exists():
        st.info("Loaded dataset from default path (new tab / new session).")
        return pd.read_csv(default_path)
    st.warning("No dataset found. Upload the CSV here or open the main page first.")
    up = st.file_uploader("Upload CSV", type=["csv"], key="mx_upload")
    if up is not None:
        return pd.read_csv(up)
    return None

df = _load_df()
if df is None:
    st.stop()

numeric = df.select_dtypes(include=["number"])
if numeric.empty:
    st.error("No numeric columns available.")
    st.stop()

method = st.selectbox("Method", ["Pearson"], index=0)
corr = numeric.corr(method=method.lower())

st.caption(f"Correlation matrix ({method})")
st.dataframe(corr.round(3), use_container_width=True)
