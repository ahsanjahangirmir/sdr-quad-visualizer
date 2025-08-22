import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

# =============================
# Helpers
# =============================

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def to_fraction(series: pd.Series) -> pd.Series:
    """Coerce a possibly-percent-encoded column to a fraction in [0,1].
    - If strings contain '%', strip and divide by 100.
    - If numeric and values mostly > 1, assume they're percents 0-100 and divide by 100.
    """
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.replace('%', '', regex=False)
        s = pd.to_numeric(s, errors='coerce')
    # Handle all-nan gracefully
    if s.notna().sum() == 0:
        return s
    # If most values look like 0..1 already, keep; else divide by 100
    gt1_ratio = (s.dropna() > 1).mean()
    if gt1_ratio > 0.5:
        s = s / 100.0
    return s


def compute_average_coe(
    df: pd.DataFrame,
    selected_vars: List[str],
    weights: Dict[str, float],
    denom: float,
) -> pd.DataFrame:
    score = 0.0
    for col in selected_vars:
        score = score + df[col] * float(weights.get(col, 0.0))
    df['Average COE Score'] = score
    denom = max(float(denom), 1e-12)
    df['Normalized COE'] = df['Average COE Score'] / denom
    return df


def compute_effort(
    df: pd.DataFrame,
    include: str,  # 'Normalized COE' | 'Average % Active' | 'Both'
    w_coe: float,
) -> pd.DataFrame:
    # Ensure Average % Active is a fraction
    if 'Average % Active' in df.columns:
        df['Average % Active (frac)'] = to_fraction(df['Average % Active'])
    else:
        df['Average % Active (frac)'] = np.nan

    if include == 'Normalized COE':
        df['Effort'] = df['Normalized COE']
    elif include == 'Average % Active':
        df['Effort'] = df['Average % Active (frac)']
    else:
        w_coe = float(np.clip(w_coe, 0.0, 1.0))
        w_act = 1.0 - w_coe
        df['Effort'] = w_coe * df['Normalized COE'] + w_act * df['Average % Active (frac)']

    df['Effort %'] = df['Effort'] * 100.0
    return df


def transform_conversion(
    df: pd.DataFrame,
    include_cr: bool,
    conv_transform: str,
    conv_norm_factor: float,
) -> pd.Series:
    if not include_cr:
        return pd.Series(np.zeros(len(df)), index=df.index)
    s = to_fraction(df['Average % Conversion Rate'])
    if conv_transform == 'Normalize':
        conv_norm_factor = max(float(conv_norm_factor), 1e-12)
        s = s / conv_norm_factor
    # else: 'No' -> unchanged fraction
    return s


def transform_attainment(
    df: pd.DataFrame,
    include_ta: bool,
    ta_transform: str,
) -> pd.Series:
    if not include_ta:
        return pd.Series(np.zeros(len(df)), index=df.index)
    s = to_fraction(df['Average % Weekly Target Point Attainment'])
    if ta_transform == 'Clip to 100':
        s = np.minimum(s, 1.0)
    elif ta_transform == 'Normalize':
        s_min, s_max = s.min(skipna=True), s.max(skipna=True)
        rng = (s_max - s_min) if pd.notna(s_min) and pd.notna(s_max) else 0.0
        if rng > 0:
            s = (s - s_min) / rng
        else:
            s = 0.0 * s
    # else: 'No' -> unchanged fraction
    return s


def compute_performance(
    df: pd.DataFrame,
    perf_choice: str,  # 'Average % Conversion Rate' | 'Average % Weekly Target Point Attainment' | 'Both'
    conv_transform: str,  # 'No' | 'Normalize'
    conv_norm_factor: float,
    ta_transform: str,    # 'No' | 'Clip to 100' | 'Normalize'
    w_cr_slider: float,
    final_transform: str,  # 'No' | 'Clip to 100' | 'Normalize'
) -> pd.DataFrame:
    include_cr = perf_choice in ('Average % Conversion Rate', 'Both')
    include_ta = perf_choice in ('Average % Weekly Target Point Attainment', 'Both')

    conv_s = transform_conversion(df, include_cr, conv_transform, conv_norm_factor)
    ta_s = transform_attainment(df, include_ta, ta_transform)

    if perf_choice == 'Average % Conversion Rate':
        perf_frac = conv_s  # already fraction-like post-transform
    elif perf_choice == 'Average % Weekly Target Point Attainment':
        perf_frac = ta_s
    else:
        w_cr = float(np.clip(w_cr_slider, 0.0, 1.0))
        w_ta = 1.0 - w_cr
        perf_frac = w_ta * ta_s + w_cr * conv_s

    perf_pct = perf_frac * 100.0

    # Final transform applied to the %-scaled series; keep output as percent 0-100 if normalized
    if final_transform == 'Clip to 100':
        perf_pct = np.minimum(perf_pct, 100.0)
    elif final_transform == 'Normalize':
        pmin, pmax = perf_pct.min(skipna=True), perf_pct.max(skipna=True)
        prng = (pmax - pmin) if pd.notna(pmin) and pd.notna(pmax) else 0.0
        if prng > 0:
            perf_pct = (perf_pct - pmin) / prng * 100.0
        else:
            perf_pct = 0.0 * perf_pct

    df['Performance %'] = perf_pct
    return df


def classify_quadrant_row(effort_pct: float, perf_pct: float, e_thr: float, p_thr: float) -> str:
    if perf_pct >= p_thr and effort_pct >= e_thr:
        return 'Superstar'
    elif perf_pct >= p_thr and effort_pct < e_thr:
        return 'Unicorn'
    elif perf_pct < p_thr and effort_pct >= e_thr:
        return 'Needs Training'
    else:
        return 'Exit'


def compute_quadrants(df: pd.DataFrame, e_thr: float, p_thr: float) -> pd.DataFrame:
    df['Quadrant'] = [
        classify_quadrant_row(e, p, e_thr, p_thr)
        for e, p in zip(df['Effort %'], df['Performance %'])
    ]
    return df


def make_scatter(df: pd.DataFrame, effort_thr: float, perf_thr: float, person_col: str) -> go.Figure:
    fig = go.Figure()

    # Points
    fig.add_trace(
        go.Scattergl(
            x=df['Effort %'],
            y=df['Performance %'],
            mode='markers',
            marker=dict(size=7, color='#00a0f9'),
            hovertemplate=(
                f"{person_col}: %{{customdata[0]}}<br>Effort %%: %{{x:.2f}}<br>Performance %%: %{{y:.2f}}<extra></extra>"
            ),
            customdata=np.c_[df[person_col]],
        )
    )

    # Threshold lines
    fig.add_shape(type="line", x0=effort_thr, x1=effort_thr, y0=df['Performance %'].min(), y1=df['Performance %'].max(),
                  line=dict(color='#ffd53e', width=2))
    fig.add_shape(type="line", x0=df['Effort %'].min(), x1=df['Effort %'].max(), y0=perf_thr, y1=perf_thr,
                  line=dict(color='#bb48dd', width=2))

    fig.update_layout(
        xaxis_title='Effort %',
        yaxis_title='Performance %',
        margin=dict(l=40, r=20, t=20, b=40),
        height=520,
    )
    return fig


def quadrant_tables(df: pd.DataFrame, person_col: str) -> Dict[str, pd.DataFrame]:
    base = df[[person_col, 'Performance %', 'Effort %', 'Quadrant']].copy()
    base['Performance %'] = base['Performance %'].round(2)
    base['Effort %'] = base['Effort %'].round(2)
    out = {}
    for q in ['Unicorn', 'Superstar', 'Needs Training', 'Exit']:
        out[q] = base[base['Quadrant'] == q][[person_col, 'Performance %', 'Effort %']]
    return out


def quadrant_stats(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    stats = {}
    for q, tbl in tables.items():
        if not tbl.empty:
            stats[q] = tbl[['Performance %', 'Effort %']].describe()
        else:
            stats[q] = pd.DataFrame()
    return stats


def settings_json_blob(config: Dict) -> str:
    return json.dumps(config, indent=2)


def copy_to_clipboard_widget(text: str, key: str = 'copy_settings'):
    import streamlit.components.v1 as components
    # Simple HTML + JS to copy provided text to clipboard
    html = f"""
    <textarea id='cfg_{key}' style='width:100%;height:180px;'>{text}</textarea>
    <button onclick="navigator.clipboard.writeText(document.getElementById('cfg_{key}').value)">Copy settings to clipboard</button>
    <style>button {{ margin-top: 6px; }}</style>
    """
    components.html(html, height=230)


# =============================
# App
# =============================

st.set_page_config(page_title='SDR Quadrant Analyzer', layout='wide')
st.title('SDR Quadrant Analyzer (June - July 2025 Data)')

# Data source
with st.sidebar:
    st.header('Data')
    default_path = 'quadrant_analysis_data.csv'
    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        path = st.text_input('Path to CSV', value=default_path)
        df = load_csv(path)

# Identify person/label column
person_col_guess = 'Person' if 'Person' in df.columns else df.columns[0]
person_col = st.sidebar.selectbox('Identifier column (for hover/table)', options=df.columns.tolist(), index=df.columns.get_loc(person_col_guess))
# =============================
# 1) Configuration Section
# =============================

st.header('Configuration')

with st.expander('Average COE Score', expanded=True):
    st.caption('Select variables and weights to compute **Average COE Score** and **Normalized COE**')

    coe_vars_all = ['Average Dials', 'Average Call Time', 'Average Emails Sent', 'Average SMS Sent']
    coe_default = coe_vars_all  # Default: Select All
    coe_selected = st.multiselect('COE Variables', options=coe_vars_all, default=coe_default)

    # Default weights
    default_w = {
        'Average Dials': 0.8,
        'Average Call Time': 0.1,
        'Average Emails Sent': 0.05,
        'Average SMS Sent': 0.05,
    }

    coe_weights = {}
    cols = st.columns(min(4, max(1, len(coe_selected)))) if coe_selected else [st]
    for i, col in enumerate(coe_selected):
        with cols[i % len(cols)]:
            coe_weights[col] = st.number_input(
                f"Weight · {col}",
                min_value=0.0,
                max_value=10.0,
                step=0.01,
                value=float(default_w.get(col, 0.0)),
            )

    coe_denominator = st.number_input('Denominator for Normalized COE', min_value=0.000001, value=1000.0, step=10.0, help='Normalized COE = Average COE Score / denominator')

with st.expander('Effort calculation', expanded=True):
    st.caption('Configure **Effort** from Normalized COE and/or Average % Active')
    effort_choice = st.radio('Include in Effort', options=['Both', 'Normalized COE', 'Average % Active'], index=0, horizontal=True)

    if effort_choice == 'Both':
        w_coe = st.slider('Weight for Normalized COE (w_coe)', 0.0, 1.0, 0.6, 0.01)
    elif effort_choice == 'Normalized COE':
        w_coe = 1.0
        st.info('Using Normalized COE only (w_coe=1, w_activity=0)')
    else:
        w_coe = 0.0
        st.info('Using Average % Active only (w_coe=0, w_activity=1)')

with st.expander('Performance calculation', expanded=True):
    st.caption('Configure **Performance %** using Conversion Rate and/or Weekly Target Point Attainment')

    perf_choice = st.radio(
        'Include in Performance',
        options=['Both', 'Average % Conversion Rate', 'Average % Weekly Target Point Attainment'],
        index=0,
        horizontal=True,
    )

    # Conversion options
    conv_transform = 'No'
    conv_norm_factor = 0.025
    if perf_choice in ('Both', 'Average % Conversion Rate'):
        st.subheader('Conversion Rate')
        conv_transform = st.selectbox('Transform (Conversion)', options=['No', 'Normalize'], index=1)
        conv_max = to_fraction(df['Average % Conversion Rate']).max(skipna=True)
        conv_max = float(conv_max) if pd.notna(conv_max) else 1.0
        # Avoid degenerate upper bound
        # conv_upper = max(conv_max, 1e-6)
        conv_upper = max(conv_max, 0.05)
        default_norm = 0.025 if 0.025 <= conv_upper else conv_upper
        conv_norm_factor = st.slider('Normalization Factor (CR)', min_value=1e-6, max_value=conv_upper, value=float(default_norm), step=1e-6, format='%.6f')

    # Attainment options
    ta_transform = 'No'
    if perf_choice in ('Both', 'Average % Weekly Target Point Attainment'):
        st.subheader('Weekly Target Point Attainment')
        ta_transform = st.selectbox('Transform (Attainment)', options=['No', 'Clip to 100', 'Normalize'], index=0)

    # Weights
    if perf_choice == 'Both':
        w_cr_slider = st.slider('Weight for Conversion (w_cr)', 0.0, 1.0, 0.5, 0.01)
    elif perf_choice == 'Average % Conversion Rate':
        w_cr_slider = 1.0
        st.info('Using Conversion only (w_cr=1, w_ta=0)')
    else:
        w_cr_slider = 0.0
        st.info('Using Attainment only (w_cr=0, w_ta=1)')

    final_transform = st.selectbox('Final transform for Performance %', options=['No', 'Clip to 100', 'Normalize'], index=1)

with st.expander('% Thresholds', expanded=True):
    col_thr1, col_thr2 = st.columns(2)
    with col_thr1:
        effort_threshold = st.number_input('% Effort Threshold', min_value=0.0, max_value=100.0, value=60.0, step=1.0)
    with col_thr2:
        performance_threshold = st.number_input('% Performance Threshold', min_value=0.0, max_value=100.0, value=60.0, step=1.0)

# Compute pipeline based on settings
if coe_selected:
    df = compute_average_coe(df, coe_selected, coe_weights, coe_denominator)
else:
    st.warning('No COE variables selected; Average COE Score will be 0.')
    df['Average COE Score'] = 0.0
    df['Normalized COE'] = 0.0

df = compute_effort(df, effort_choice, w_coe)

df = compute_performance(
    df,
    perf_choice=perf_choice,
    conv_transform=conv_transform,
    conv_norm_factor=conv_norm_factor,
    ta_transform=ta_transform,
    w_cr_slider=w_cr_slider,
    final_transform=final_transform,
)

df = compute_quadrants(df, effort_threshold, performance_threshold)

# Settings JSON + copy/download
settings = {
    'coe': {
        'selected_vars': coe_selected,
        'weights': coe_weights,
        'denominator': coe_denominator,
    },
    'effort': {
        'include': effort_choice,
        'w_coe': w_coe,
    },
    'performance': {
        'choice': perf_choice,
        'conv_transform': conv_transform,
        'conv_norm_factor': conv_norm_factor,
        'ta_transform': ta_transform,
        'w_cr': w_cr_slider,
        'final_transform': final_transform,
    },
    'thresholds': {
        'effort_threshold_pct': effort_threshold,
        'performance_threshold_pct': performance_threshold,
    },
    'person_col': person_col,
}

st.subheader('Copy settings as JSON')
blob = settings_json_blob(settings)
copy_to_clipboard_widget(blob, key='settings_json')
st.download_button('Download settings.json', data=blob, file_name='settings.json', mime='application/json')

# =============================
# 2) Dataset Information Section
# =============================

st.header('Dataset Information')
st.dataframe(df.describe(include='all'))

# =============================
# 3) Chart Section
# =============================

# Make the raw dataframe available to other pages
st.session_state['raw_df'] = df.copy()

st.header('Performance % vs Effort % Graph')

# Right-aligned "View Raw Dataset" link
r1, r2 = st.columns([1, 0.15])
with r2:
    try:
        st.page_link(
            "pages/01_Raw_Dataset.py",
            label="View Raw Dataset",
            icon=":material/grid_on:",
            new_tab=True,            
        )
    except TypeError:
        # Older Streamlit without new_tab support -> still adds a correct link.
        st.page_link(
            "pages/01_Raw_Dataset.py",
            label="View Raw Dataset",
            icon=":material/grid_on:",
        )

fig = make_scatter(df, effort_threshold, performance_threshold, person_col)
st.plotly_chart(fig, use_container_width=True, theme=None)

# =============================
# 4) Summary Section
# =============================

st.header('SDRs by Quadrant')
qtables = quadrant_tables(df, person_col)
col_u, col_s, col_n, col_e = st.columns(4)
with col_u:
    st.subheader('Unicorn')
    st.dataframe(qtables['Unicorn'])
with col_s:
    st.subheader('Superstar')
    st.dataframe(qtables['Superstar'])
with col_n:
    st.subheader('Needs Training')
    st.dataframe(qtables['Needs Training'])
with col_e:
    st.subheader('Exit')
    st.dataframe(qtables['Exit'])

# =============================
# 5) Summary Information Section
# =============================

st.header('Descriptive Stats by Quadrant')
stats = quadrant_stats(qtables)
col_u2, col_s2, col_n2, col_e2 = st.columns(4)
with col_u2:
    st.subheader('Unicorn')
    if not stats['Unicorn'].empty:
        st.dataframe(stats['Unicorn'])
    else:
        st.caption('No rows')
with col_s2:
    st.subheader('Superstar')
    if not stats['Superstar'].empty:
        st.dataframe(stats['Superstar'])
    else:
        st.caption('No rows')
with col_n2:
    st.subheader('Needs Training')
    if not stats['Needs Training'].empty:
        st.dataframe(stats['Needs Training'])
    else:
        st.caption('No rows')
with col_e2:
    st.subheader('Exit')
    if not stats['Exit'].empty:
        st.dataframe(stats['Exit'])
    else:
        st.caption('No rows')

st.caption('© Quadrant Analyzer — Streamlit + Plotly')
