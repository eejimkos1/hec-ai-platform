"""HEC corporate theme — CSS injection and styled component helpers."""
import streamlit as st
from app.config import COLORS

def inject_hec_css():
    """Inject HEC corporate CSS. Call once at the top of every page."""
    st.markdown('''<style>
    /* --- Sidebar: dark navy --- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1B2A 0%, #1B2838 100%) !important;
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
        padding-top: 1rem;
    }
    h1 { color: #0D1B2A !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    h2, h3 { color: #0D1B2A !important; font-weight: 600 !important; }

    /* --- Metric cards --- */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E4E8;
        border-left: 4px solid #1B9AAA;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(13,27,42,0.06);
        transition: box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 16px rgba(13,27,42,0.12);
    }
    [data-testid="stMetric"] label {
        color: #546E7A !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 500 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0D1B2A !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 2px solid #E0E4E8;
        background: linear-gradient(180deg, #F8FAFB 0%, #FFFFFF 100%);
        border-radius: 8px 8px 0 0;
        padding: 4px 8px 0;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 500;
        color: #546E7A;
        padding: 10px 22px;
        font-size: 0.88rem;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #0D1B2A !important;
        border-bottom: 3px solid #1B9AAA;
        font-weight: 700;
        background: white;
        border-radius: 6px 6px 0 0;
    }

    /* --- Buttons --- */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #E0E4E8;
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stFormSubmitButton"] {
        background: linear-gradient(135deg, #0D1B2A 0%, #1B2838 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(13,27,42,0.2);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1B9AAA 0%, #148F9E 100%) !important;
        box-shadow: 0 4px 12px rgba(27,154,170,0.3);
    }

    /* --- Expanders --- */
    [data-testid="stExpander"] {
        border: 1px solid #E0E4E8;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(13,27,42,0.04);
    }
    [data-testid="stExpander"] summary {
        font-weight: 500;
    }

    /* --- Data frames --- */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(13,27,42,0.06);
    }

    /* --- Dividers --- */
    hr { border-color: #E0E4E8 !important; }

    /* --- Multiselect & selectbox --- */
    .stMultiSelect, .stSelectbox {
        margin-bottom: 0.5rem;
    }
    </style>''', unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, icon: str = ""):
    """Render a professional gradient header banner for a page."""
    st.markdown(f'''
    <div style="
        background: linear-gradient(135deg, #0D1B2A 0%, #1B3A4B 50%, #1B9AAA 100%);
        border-radius: 12px;
        padding: 32px 36px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(13,27,42,0.15);
    ">
        <div style="font-size: 2rem; margin-bottom: 4px;">{icon}</div>
        <h1 style="color: white !important; margin: 0 0 8px 0; font-size: 1.8rem; font-weight: 700;">{title}</h1>
        <p style="color: rgba(255,255,255,0.85); margin: 0; font-size: 1rem; font-weight: 400;">{subtitle}</p>
    </div>
    ''', unsafe_allow_html=True)


def render_section_header(title: str, icon: str = "", description: str = ""):
    """Render a styled section header with optional icon and description."""
    desc_html = f'<p style="color:#546E7A; margin:4px 0 0; font-size:0.9rem;">{description}</p>' if description else ""
    st.markdown(f'''
    <div style="margin: 24px 0 16px; padding-bottom: 8px; border-bottom: 2px solid #E0E4E8;">
        <h3 style="margin:0; color:#0D1B2A; font-weight:600;">
            {(icon + " ") if icon else ""}{title}
        </h3>
        {desc_html}
    </div>
    ''', unsafe_allow_html=True)


def render_money_callout(title: str, amount_eur: float, description: str):
    """Render a highlighted financial callout box with teal left border."""
    st.markdown(f'''
    <div style="background: linear-gradient(135deg, #F0FAFA 0%, #E8F8F9 100%);
                border-left:5px solid #1B9AAA; border-radius:8px;
                padding:22px 28px; margin:16px 0;
                box-shadow: 0 2px 8px rgba(27,154,170,0.1);">
        <div style="font-size:0.82rem; color:#546E7A; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:6px; font-weight:500;">{title}</div>
        <div style="font-size:2.2rem; font-weight:700; color:#0D1B2A; letter-spacing:-0.5px;">€{amount_eur:,.0f}</div>
        <div style="font-size:0.88rem; color:#546E7A; margin-top:8px;">{description}</div>
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


def render_kpi_card(label: str, value: str, icon: str = "", color: str = "#1B9AAA"):
    """Render a single styled KPI card with custom color accent."""
    st.markdown(f'''
    <div style="background:#FFFFFF; border:1px solid #E0E4E8; border-left:4px solid {color};
                border-radius:8px; padding:16px 20px; box-shadow:0 2px 8px rgba(13,27,42,0.06);
                text-align:center;">
        <div style="font-size:1.4rem; margin-bottom:4px;">{icon}</div>
        <div style="font-size:1.6rem; font-weight:700; color:#0D1B2A;">{value}</div>
        <div style="font-size:0.78rem; color:#546E7A; text-transform:uppercase; letter-spacing:0.5px; margin-top:4px;">{label}</div>
    </div>
    ''', unsafe_allow_html=True)


def apply_plotly_theme(fig):
    """Apply HEC corporate theme to a Plotly figure."""
    from app.config import PLOTLY_LAYOUT
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        xaxis=dict(gridcolor="#E0E4E8", zerolinecolor="#E0E4E8"),
        yaxis=dict(gridcolor="#E0E4E8", zerolinecolor="#E0E4E8"),
    )
    return fig
