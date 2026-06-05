"""HEC corporate theme — CSS injection and styled component helpers."""
import streamlit as st
from app.config import COLORS

def inject_hec_css():
    """Inject HEC corporate CSS. Call once at the top of every page."""
    st.markdown('''<style>
    /* --- Sidebar: dark navy --- */
    [data-testid="stSidebar"] {
        background-color: #0D1B2A !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"],
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stCaption p {
        color: #B0BEC5 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }
    [data-testid="stSidebar"] a {
        color: #4DB6AC !important;
    }

    /* --- Typography --- */
    .main .block-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    }
    h1 { color: #0D1B2A !important; font-weight: 700 !important; }
    h2, h3 { color: #0D1B2A !important; font-weight: 600 !important; }

    /* --- Metric cards --- */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E4E8;
        border-left: 4px solid #1B9AAA;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(13,27,42,0.08);
    }
    [data-testid="stMetric"] label {
        color: #546E7A !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0D1B2A !important;
        font-weight: 700 !important;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 2px solid #E0E4E8;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 500;
        color: #546E7A;
        padding: 8px 20px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #1B9AAA !important;
        border-bottom: 3px solid #1B9AAA;
        font-weight: 600;
    }

    /* --- Buttons --- */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid #E0E4E8;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stFormSubmitButton"] {
        background-color: #0D1B2A !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1B9AAA !important;
    }

    /* --- Expanders --- */
    [data-testid="stExpander"] {
        border: 1px solid #E0E4E8;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(13,27,42,0.04);
    }

    /* --- Dividers --- */
    hr { border-color: #E0E4E8 !important; }
    </style>''', unsafe_allow_html=True)


def render_money_callout(title: str, amount_eur: float, description: str):
    """Render a highlighted financial callout box with teal left border."""
    st.markdown(f'''
    <div style="background:#F0FAFA; border-left:5px solid #1B9AAA; border-radius:8px; padding:20px 24px; margin:16px 0;">
        <div style="font-size:0.85rem; color:#546E7A; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">{title}</div>
        <div style="font-size:2rem; font-weight:700; color:#0D1B2A;">€{amount_eur:,.0f}</div>
        <div style="font-size:0.9rem; color:#546E7A; margin-top:6px;">{description}</div>
    </div>
    ''', unsafe_allow_html=True)


def render_financial_kpi_row(metrics: list[dict]):
    """Render a row of financial KPI cards.

    Args:
        metrics: list of {"label": str, "value": str, "delta": str|None}
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(m["label"], m["value"], delta=m.get("delta"))


def apply_plotly_theme(fig):
    """Apply HEC corporate theme to a Plotly figure."""
    from app.config import PLOTLY_LAYOUT
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        xaxis=dict(gridcolor="#E0E4E8", zerolinecolor="#E0E4E8"),
        yaxis=dict(gridcolor="#E0E4E8", zerolinecolor="#E0E4E8"),
    )
    return fig
