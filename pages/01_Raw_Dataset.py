# pages/01_Raw_Dataset.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Raw Dataset", layout="wide")
st.title("Raw Dataset")

if "raw_df" in st.session_state:
    raw_df = st.session_state["raw_df"]
    st.dataframe(raw_df, use_container_width=True)
    st.download_button(
        "Download as CSV",
        data=raw_df.to_csv(index=False),
        file_name="raw_dataset.csv",
        mime="text/csv",
    )
else:
    st.warning("No dataset found in session. Open the main page first so the data is loaded.")
