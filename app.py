import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from urllib.parse import urlencode
import streamlit.components.v1 as components


BUCKET_COLORS = {
    "Top 20%": "#22c55e",        # green
    "Upper-Mid 30%": "#3b82f6",  # blue
    "Lower-Mid 30%": "#f59e0b",  # amber
    "Bottom 20%": "#f43f5e",     # pinkish red
}

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

    if 'Bucket' in df.columns:
        # one trace per bucket to get a legend and distinct colors
        for bucket, sub in df.groupby('Bucket', dropna=False):
            name = str(bucket) if pd.notna(bucket) else 'Unspecified'
            color = BUCKET_COLORS.get(bucket, '#94a3b8')  # fallback gray
            fig.add_trace(
                go.Scattergl(
                    x=sub['Effort %'],
                    y=sub['Performance %'],
                    mode='markers',
                    name=name,
                    marker=dict(size=7, color=color),
                    hovertemplate=(
                        f"{person_col}: %{{customdata[0]}}"
                        "<br>Bucket: " + name +
                        "<br>Effort %%: %{x:.2f}"
                        "<br>Performance %%: %{y:.2f}<extra></extra>"
                    ),
                    customdata=np.c_[sub[person_col]],
                )
            )
    else:
        # fallback (previous single-trace behavior)
        fig.add_trace(
            go.Scattergl(
                x=df['Effort %'],
                y=df['Performance %'],
                mode='markers',
                marker=dict(size=7, color='#00a0f9'),
                hovertemplate=(
                    f"{person_col}: %{{customdata[0]}}"
                    "<br>Effort %%: %{x:.2f}"
                    "<br>Performance %%: %{y:.2f}<extra></extra>"
                ),
                customdata=np.c_[df[person_col]],
            )
        )

    # Threshold lines
    fig.add_shape(
        type="line",
        x0=effort_thr, x1=effort_thr,
        y0=df['Performance %'].min(), y1=df['Performance %'].max(),
        line=dict(color='#ffd53e', width=2)
    )
    fig.add_shape(
        type="line",
        x0=df['Effort %'].min(), x1=df['Effort %'].max(),
        y0=perf_thr, y1=perf_thr,
        line=dict(color='#bb48dd', width=2)
    )

    fig.update_layout(
        xaxis_title='Effort %',
        yaxis_title='Performance %',
        margin=dict(l=40, r=20, t=20, b=40),
        height=720,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
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

# -----------------------------
# Query Param helpers
# -----------------------------

# Short codes for COE vars (stable URL keys)
COE_LONG2SHORT = {
    'Average Dials': 'dials',
    'Average Call Time': 'call',
    'Average Emails Sent': 'emails',
    'Average SMS Sent': 'sms',
}
COE_SHORT2LONG = {v: k for k, v in COE_LONG2SHORT.items()}

TF_OUT = {'No': 'no', 'Normalize': 'norm', 'Clip to 100': 'clip'}
TF_IN  = {v: k for k, v in TF_OUT.items()}

EFFORT_OUT = {'Both': 'both', 'Normalized COE': 'coe', 'Average % Active': 'active'}
EFFORT_IN  = {v: k for k, v in EFFORT_OUT.items()}

PERF_OUT = {'Both': 'both', 'Average % Conversion Rate': 'cr', 'Average % Weekly Target Point Attainment': 'ta'}
PERF_IN  = {v: k for k, v in PERF_OUT.items()}

def _qp_raw() -> dict:
    """Return current query params as {k:[v,...]} for both new/old Streamlit APIs."""
    if hasattr(st, "query_params"):
        q = st.query_params
        try:
            d = dict(q)  # may already be dict[str,str]
        except Exception:
            d = q.to_dict()  # best effort
        # Normalize to list form
        out = {}
        for k, v in d.items():
            out[k] = v if isinstance(v, list) else [v]
        return out
    else:
        # legacy
        return st.experimental_get_query_params()

def _qp_get_str(q: dict, key: str, default: str | None = None) -> str | None:
    v = q.get(key)
    if not v:
        return default
    return v[0] if isinstance(v, list) else v

def _qp_get_float(q: dict, key: str, default: float | None = None) -> float | None:
    s = _qp_get_str(q, key, None)
    if s is None:
        return default
    try:
        return float(s)
    except:
        return default

def _parse_csv_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(',') if x.strip()]

def settings_to_query_params(settings: dict) -> dict[str, str]:
    """Flatten your `settings` dict to URL-safe key=val strings."""
    coe_vars_short = [COE_LONG2SHORT[v] for v in settings['coe']['selected_vars'] if v in COE_LONG2SHORT]
    weights = settings['coe']['weights']
    params = {
        'person': settings['person_col'],
        'coe_vars': ','.join(coe_vars_short),
        'w_dials': f"{float(weights.get('Average Dials', 0.0)):.6g}",
        'w_call': f"{float(weights.get('Average Call Time', 0.0)):.6g}",
        'w_emails': f"{float(weights.get('Average Emails Sent', 0.0)):.6g}",
        'w_sms': f"{float(weights.get('Average SMS Sent', 0.0)):.6g}",
        'coe_denom': f"{float(settings['coe']['denominator']):.6g}",
        'effort': EFFORT_OUT[settings['effort']['include']],
        'w_coe': f"{float(settings['effort']['w_coe']):.6g}",
        'perf': PERF_OUT[settings['performance']['choice']],
        'cr_tf': TF_OUT[settings['performance']['conv_transform']],
        'cr_norm': f"{float(settings['performance']['conv_norm_factor']):.6g}",
        'ta_tf': TF_OUT[settings['performance']['ta_transform']],
        'w_cr': f"{float(settings['performance']['w_cr']):.6g}",
        'final_tf': TF_OUT[settings['performance']['final_transform']],
        'e_thr': f"{float(settings['thresholds']['effort_threshold_pct']):.6g}",
        'p_thr': f"{float(settings['thresholds']['performance_threshold_pct']):.6g}",
    }
    return params


def _current_query_as_strdict() -> dict[str,str]:
    q = _qp_raw()
    out = {}
    for k, v in q.items():
        out[k] = v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else str(v))
    return out

def set_query_params_if_changed(params: dict[str, str]) -> None:
    """Only update the URL if needed (prevents infinite reruns)."""
    cur = _current_query_as_strdict()
    if cur != params:
        if hasattr(st, "query_params"):
            try:
                st.query_params.clear()
                st.query_params.update(params)
            except Exception:
                pass
        else:
            st.experimental_set_query_params(**params)

def copy_share_url_widget(params: dict) -> None:
    """Renders a 'Copy Chart URL' button that builds an absolute URL in the browser and copies it."""
    js_payload = json.dumps(params)
    html = f"""
    <div style='display:flex;gap:8px;align-items:center;margin-top:6px;'>
      <input id='share_url_input' style='flex:1;padding:6px 8px;border-radius:6px;border:1px solid #d0d4da;' readonly />
      <button id='copy_url_btn' style='padding:6px 10px;border:1px solid #d0d4da;border-radius:6px;cursor:pointer;'>Copy Chart URL</button>
    </div>
    <script>
      (function() {{
        const params = {js_payload};
        const usp = new URLSearchParams(params);
        const url = window.location.origin + window.location.pathname + "?" + usp.toString();
        const input = document.getElementById('share_url_input');
        const btn = document.getElementById('copy_url_btn');
        input.value = url;
        btn.onclick = () => navigator.clipboard.writeText(url);
      }})();
    </script>
    """
    components.html(html, height=60)


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

# --- Read query params to prefill defaults ---
_q = _qp_raw()

# person column default via ?person=
person_options = df.columns.tolist()
qp_person = _qp_get_str(_q, 'person', None)
if qp_person in person_options:
    person_default_index = person_options.index(qp_person)
else:
    # your existing guess fallback
    person_col_guess = 'Person' if 'Person' in df.columns else df.columns[0]
    person_default_index = person_options.index(person_col_guess)

# Identify person/label column (prefilled from query params if present)
person_col = st.sidebar.selectbox(
    'Identifier column (for hover/table)',
    options=person_options,
    index=person_default_index
)
# =============================
# 1) Configuration Section
# =============================

st.header('Configuration')

with st.expander('Average COE Score', expanded=True):
    st.caption('Select variables and weights to compute **Average COE Score** and **Normalized COE**')

    coe_vars_all = ['Average Dials', 'Average Call Time', 'Average Emails Sent', 'Average SMS Sent']

    # From query params: ?coe_vars=dials,call,emails,sms
    qp_coe_vars = _parse_csv_list(_qp_get_str(_q, 'coe_vars', None))
    coe_default = [COE_SHORT2LONG[s] for s in qp_coe_vars if s in COE_SHORT2LONG] or coe_vars_all

    coe_selected = st.multiselect('COE Variables', options=coe_vars_all, default=coe_default)

    # Default weights (override by query params if present)
    default_w = {
        'Average Dials': float(_qp_get_float(_q, 'w_dials', 0.8)),
        'Average Call Time': float(_qp_get_float(_q, 'w_call', 0.1)),
        'Average Emails Sent': float(_qp_get_float(_q, 'w_emails', 0.05)),
        'Average SMS Sent': float(_qp_get_float(_q, 'w_sms', 0.05)),
    }

    coe_weights = {}
    cols = st.columns(min(4, max(1, len(coe_selected)))) if coe_selected else [st]
    for i, col in enumerate(coe_selected):
        with cols[i % len(cols)]:
            coe_weights[col] = st.number_input(
                f"Weight · {col}",
                min_value=0.0, max_value=10.0, step=0.01,
                value=float(default_w.get(col, 0.0)),
            )

    coe_denominator = st.number_input(
        'Denominator for Normalized COE',
        min_value=0.000001,
        value=float(_qp_get_float(_q, 'coe_denom', 1000.0)),
        step=10.0,
        help='Normalized COE = Average COE Score / denominator'
    )


with st.expander('Effort calculation', expanded=True):
    st.caption('Configure **Effort** from Normalized COE and/or Average % Active')
    
    effort_default = EFFORT_IN.get(_qp_get_str(_q, 'effort', 'both'), 'Both')
    effort_choice = st.radio('Include in Effort',
                         options=['Both', 'Normalized COE', 'Average % Active'],
                         index=['Both','Normalized COE','Average % Active'].index(effort_default),
                         horizontal=True)

    if effort_choice == 'Both':
        w_coe_default = float(_qp_get_float(_q, 'w_coe', 0.6))
        w_coe = st.slider('Weight for Normalized COE (w_coe)', 0.0, 1.0, w_coe_default, 0.01)
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
        index=['Both','Average % Conversion Rate','Average % Weekly Target Point Attainment'].index(
            PERF_IN.get(_qp_get_str(_q, 'perf', 'both'), 'Both')
        ),
        horizontal=True,
    )

    # Conversion options
    conv_transform = 'No'
    conv_norm_factor = 0.025
    if perf_choice in ('Both', 'Average % Conversion Rate'):
        st.subheader('Conversion Rate')
        conv_transform = st.selectbox('Transform (Conversion)', options=['No', 'Normalize'],
                                    index=['No','Normalize'].index(TF_IN.get(_qp_get_str(_q, 'cr_tf', 'norm'), 'Normalize')))

        conv_max = to_fraction(df['Average % Conversion Rate']).max(skipna=True)
        conv_max = float(conv_max) if pd.notna(conv_max) else 1.0
        conv_upper = max(conv_max, 0.05)
        default_norm = float(_qp_get_float(_q, 'cr_norm', 0.025))
        default_norm = default_norm if default_norm <= conv_upper else conv_upper
        conv_norm_factor = st.slider('Normalization Factor (CR)', min_value=1e-6, max_value=conv_upper,
                                    value=float(default_norm), step=1e-6, format='%.6f')

    # Attainment options
    ta_transform = 'No'
    if perf_choice in ('Both', 'Average % Weekly Target Point Attainment'):
        st.subheader('Weekly Target Point Attainment')
        ta_transform = st.selectbox('Transform (Attainment)',
                            options=['No', 'Clip to 100', 'Normalize'],
                            index=['No','Clip to 100','Normalize'].index(
                                TF_IN.get(_qp_get_str(_q, 'ta_tf', 'no'), 'No')
                            ))

    # Weights
    if perf_choice == 'Both':
        w_cr_slider = st.slider('Weight for Conversion (w_cr)', 0.0, 1.0,
                        float(_qp_get_float(_q, 'w_cr', 0.5)), 0.01)
    elif perf_choice == 'Average % Conversion Rate':
        w_cr_slider = 1.0
        st.info('Using Conversion only (w_cr=1, w_ta=0)')
    else:
        w_cr_slider = 0.0
        st.info('Using Attainment only (w_cr=0, w_ta=1)')

    final_transform = st.selectbox('Final transform for Performance %',
                               options=['No', 'Clip to 100', 'Normalize'],
                               index=['No','Clip to 100','Normalize'].index(
                                   TF_IN.get(_qp_get_str(_q, 'final_tf', 'clip'), 'Clip to 100')
                               ))

with st.expander('% Thresholds', expanded=True):
    col_thr1, col_thr2 = st.columns(2)
    with col_thr1:
        effort_threshold = st.number_input('% Effort Threshold', min_value=0.0, max_value=100.0,
                                   value=float(_qp_get_float(_q, 'e_thr', 60.0)), step=1.0)
    with col_thr2:
        performance_threshold = st.number_input('% Performance Threshold', min_value=0.0, max_value=100.0,
                                        value=float(_qp_get_float(_q, 'p_thr', 60.0)), step=1.0)

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

# --- Sync URL with current settings & render share URL control ---
qp_params = settings_to_query_params(settings)
set_query_params_if_changed(qp_params)

st.subheader('Copy settings as JSON')
blob = settings_json_blob(settings)
copy_to_clipboard_widget(blob, key='settings_json')
st.download_button('Download settings.json', data=blob, file_name='settings.json', mime='application/json')

# Copy a unique chart URL with current settings embedded
st.caption('Copy chart URL')
qs = urlencode(qp_params)
st.code('https://sdr-quad-visualizer-3xjc29ejzxihpcshuqkyvu.streamlit.app/' + '?' + qs, language='text')  # quick visual of the querystring
# copy_share_url_widget(qp_params)    # button copies full absolute URL

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

st.markdown(
    f"""
    <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin: 0.25rem 0 0.5rem 0;">
      <div style="display:flex; align-items:center; gap:6px;">
        <span style="width:14px;height:14px;background:#ffd53e;border-radius:3px;display:inline-block;"></span>
        <span><b>Effort Threshold:</b> {effort_threshold:.2f}%</span>
        <span>|</span>
      </div>
      <div style="display:flex; align-items:center; gap:6px;">
        <span style="width:14px;height:14px;background:#bb48dd;border-radius:3px;display:inline-block;"></span>
        <span><b>Performance Threshold:</b> {performance_threshold:.2f}%</span>
        <span>|</span>
      </div>
      <div style="display:flex; align-items:center; gap:6px;">
        <span><b>CR Normalization Factor:</b> {conv_norm_factor:.4f}</span>
        <span>|</span>
      </div>
      <div style="display:flex; align-items:center; gap:6px;">
        <span><b>Conversion Rate Weightage (Performance):</b> {w_cr_slider:.2f}</span>
        <span>|</span>
      </div>
      <div style="display:flex; align-items:center; gap:6px;">
        <span><b>Attainment Weightage (Performance):</b> {(1.0 - w_cr_slider):.2f}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
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

