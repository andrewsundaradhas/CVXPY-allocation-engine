from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.express as px  # noqa: F401  (kept for downstream polish hooks)
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots  # noqa: F401  (kept for downstream polish hooks)

from allocator.engine import AllocationRequest, run_allocation

st.set_page_config(
    page_title="AlphaAlloc — Portfolio Engine",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Design tokens (mirror styles.css from the AlphaAlloc design bundle) ===
COLORS = {
    "bg":         "#0A0D12",
    "bg_2":       "#0E1218",
    "surface":    "#11161E",
    "surface_2":  "#171D27",
    "surface_3":  "#1B2230",
    "border":     "#1E2530",
    "border_soft": "#161C26",
    "text":       "#ECEFF5",
    "text_muted": "#8C95A8",
    "text_dim":   "#5A6273",
    "up":         "#16C784",
    "down":       "#EA3943",
    "warn":       "#F0B90B",
    "accent":     "#5EEAD4",
    "accent_rgb": "94, 234, 212",
    "accent_dim": "rgba(94, 234, 212, 0.13)",
    "accent_glow": "rgba(94, 234, 212, 0.30)",
}

FONTS = {
    "ui":  "'Inter', -apple-system, system-ui, sans-serif",
    "num": "'JetBrains Mono', 'SF Mono', Consolas, monospace",
}

# Static portion of the design CSS — verbatim from styles.css where compatible
# with Streamlit's DOM. Streamlit-chrome overrides are inlined separately so
# they pick up the same custom properties.
_STATIC_CSS = r"""
* { box-sizing: border-box; }

.stApp {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', -apple-system, system-ui, sans-serif;
  font-size: var(--base-font);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  letter-spacing: -0.005em;
}

/* Ambient backdrop — barely-there atmospheric glow */
.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    radial-gradient(900px 500px at 18% -8%, rgba(94,234,212,0.055), transparent 60%),
    radial-gradient(1100px 700px at 110% 110%, rgba(120,140,255,0.045), transparent 55%),
    radial-gradient(700px 400px at 50% 120%, rgba(94,234,212,0.025), transparent 60%);
  z-index: 0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container {
  padding-top: 1.1rem;
  padding-bottom: 2rem;
  max-width: 1600px;
  position: relative;
  z-index: 1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(255,255,255,0.012), transparent 35%),
    var(--surface);
  border-right: 1px solid var(--border);
  box-shadow: inset -1px 0 0 rgba(255,255,255,0.015);
}
section[data-testid="stSidebar"] > div { padding-top: 0.6rem; }

.tabular { font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace; font-feature-settings: "tnum"; }

/* === Sidebar bits === */
.side-title {
  font-weight: 700;
  font-size: 1.02rem;
  color: var(--text);
  padding: 0.3rem 0 1.1rem;
  letter-spacing: -0.01em;
  display: flex;
  align-items: center;
  gap: 8px;
}
.side-title .mark {
  color: var(--accent);
  text-shadow: 0 0 14px var(--accent-glow);
  font-size: 1.1rem;
  display: inline-block;
  animation: spin-slow 18s linear infinite;
  transform-origin: center;
}
@keyframes spin-slow {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
.side-section {
  font-size: 0.66rem;
  font-weight: 600;
  color: var(--text-muted);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin: 1.1rem 0 0.45rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.side-section::after {
  content: "";
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* Streamlit widget overrides inside sidebar */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p {
  color: var(--text-muted);
  font-size: 0.74rem;
  font-weight: 500;
  letter-spacing: 0.04em;
}
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] input[type="text"],
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
  border-radius: 7px !important;
  transition: border-color .18s ease, box-shadow .18s ease;
}
section[data-testid="stSidebar"] textarea:focus,
section[data-testid="stSidebar"] input[type="text"]:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(94,234,212,0.10) !important;
  background: var(--surface-3) !important;
}

/* Streamlit slider — accent track + glowing thumb */
[data-baseweb="slider"] [role="slider"] {
  background: var(--accent) !important;
  border: 2px solid var(--bg) !important;
  box-shadow: 0 0 0 0 var(--accent-glow);
  transition: box-shadow .15s ease, transform .15s ease;
}
[data-baseweb="slider"] [role="slider"]:hover {
  box-shadow: 0 0 0 5px rgba(94,234,212,0.18);
  transform: scale(1.08);
}
[data-baseweb="slider"] > div > div { background: var(--accent) !important; }

/* Toggle */
[data-testid="stCheckbox"] label > div[role="checkbox"][aria-checked="true"] {
  background: var(--accent-dim) !important;
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(94,234,212,0.08) !important;
}
[data-testid="stCheckbox"] label > div[role="checkbox"][aria-checked="true"] > div {
  background: var(--accent) !important;
  box-shadow: 0 0 8px var(--accent-glow);
}

/* Segmented control (density) */
[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
  background: linear-gradient(180deg, rgba(94,234,212,0.22), rgba(94,234,212,0.10)) !important;
  color: var(--accent) !important;
  box-shadow: inset 0 0 0 1px rgba(94,234,212,0.35), 0 0 12px -4px rgba(94,234,212,0.5) !important;
}

/* === Glass-CTA Optimize button === */
.stButton > button {
  width: 100%;
  background:
    radial-gradient(120% 180% at 0% 0%, rgba(255,255,255,0.18), rgba(255,255,255,0) 55%),
    linear-gradient(180deg, #5BE9D2 0%, #3FD0BA 100%) !important;
  color: #03110D !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.01em !important;
  padding: 0.78rem 1.1rem !important;
  border-radius: 10px !important;
  cursor: pointer;
  transition: transform .18s cubic-bezier(.2,.7,.3,1), box-shadow .18s ease, filter .18s ease !important;
  font-family: inherit !important;
  position: relative;
  overflow: hidden;
  box-shadow:
    0 1px 0 rgba(255,255,255,0.45) inset,
    0 -10px 18px -10px rgba(0,0,0,0.30) inset,
    0 6px 14px -6px rgba(94,234,212,0.55),
    0 18px 40px -18px rgba(94,234,212,0.6),
    0 0 0 1px rgba(94,234,212,0.35) !important;
}
.stButton > button:hover {
  transform: translateY(-1px);
  filter: brightness(1.04) saturate(1.05);
  box-shadow:
    0 1px 0 rgba(255,255,255,0.5) inset,
    0 -10px 18px -10px rgba(0,0,0,0.32) inset,
    0 10px 22px -8px rgba(94,234,212,0.7),
    0 24px 48px -18px rgba(94,234,212,0.7),
    0 0 0 1px rgba(94,234,212,0.55) !important;
}
.stButton > button:active {
  transform: translateY(0);
  filter: brightness(0.97);
}

/* === Top bar === */
.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.45rem 0 1.05rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.25rem;
  position: relative;
}
.topbar::after {
  content: "";
  position: absolute;
  bottom: -1px; left: 0;
  width: 80px; height: 1px;
  background: linear-gradient(90deg, var(--accent), transparent);
}
.brand { font-weight: 700; font-size: 1.05rem; letter-spacing: -0.015em; display: flex; align-items: center; gap: 8px; }
.brand .mark {
  color: var(--accent);
  font-size: 1.1rem;
  text-shadow: 0 0 14px var(--accent-glow);
}
.brand .sub { color: var(--text-dim); font-weight: 400; margin-left: 4px; font-size: 0.92rem; letter-spacing: 0; }
.topbar-right { display: flex; align-items: center; gap: 14px; }
.tickers-line {
  color: var(--text-muted);
  font-size: 0.75rem;
  font-family: 'JetBrains Mono', monospace;
  max-width: 540px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  letter-spacing: 0.02em;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 4px 11px;
  border-radius: 999px;
  font-size: 0.66rem;
  font-weight: 600;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  border: 1px solid transparent;
}
.pill.ok {
  background: rgba(22,199,132,0.10);
  color: var(--up);
  border-color: rgba(22,199,132,0.22);
  box-shadow: 0 0 18px -6px rgba(22,199,132,0.4);
}
.pill.idle { background: rgba(90,98,115,0.14); color: var(--text-muted); border-color: rgba(90,98,115,0.25); }
.pill .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; box-shadow: 0 0 0 0 currentColor; animation: pulse 2s infinite; }
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(22,199,132,0.5); }
  70% { box-shadow: 0 0 0 6px rgba(22,199,132,0); }
  100% { box-shadow: 0 0 0 0 rgba(22,199,132,0); }
}

/* === KPI === */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--gap); margin-bottom: var(--gap); }
.kpi {
  background: linear-gradient(180deg, rgba(255,255,255,0.018), transparent 40%), var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: var(--pad-kpi);
  position: relative;
  overflow: hidden;
  box-shadow: var(--shadow-card);
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
.kpi:hover {
  transform: translateY(-1px);
  border-color: rgba(94,234,212,0.18);
  box-shadow: var(--shadow-card), 0 0 28px -10px rgba(94,234,212,0.25);
}
.kpi::before {
  content: ""; position: absolute; top: 0; left: 0;
  width: 2px; height: 100%;
  background: linear-gradient(180deg, var(--accent), rgba(94,234,212,0.1));
  box-shadow: 0 0 12px var(--accent-glow);
}
.kpi::after {
  content: "";
  position: absolute;
  top: -40%; right: -40%;
  width: 120px; height: 120px;
  background: radial-gradient(closest-side, rgba(94,234,212,0.08), transparent 70%);
  pointer-events: none;
}
.kpi-label {
  font-size: 0.66rem; font-weight: 600; color: var(--text-muted);
  letter-spacing: 0.12em; text-transform: uppercase;
}
.kpi-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: var(--kpi-value); font-weight: 600; color: var(--text);
  line-height: 1.15; margin: 0.35rem 0 0.2rem;
  font-feature-settings: "tnum";
  letter-spacing: -0.015em;
}
.kpi-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: var(--text-dim); letter-spacing: 0.01em; }
.kpi-sub.up { color: var(--up); }
.kpi-sub.down { color: var(--down); }

/* === Card ===
   Two paths: (a) my custom HTML .card class (used inside main flow), and
   (b) Streamlit's st.container(border=True) wrapper, which is what actually
   holds chart + title in DOM. Both must look identical. */
.card,
[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.012), transparent 30%), var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: var(--pad-card) !important;
  margin-bottom: var(--gap) !important;
  box-shadow: var(--shadow-card) !important;
  transition: border-color .25s ease, box-shadow .25s ease !important;
  position: relative;
}
.card:hover,
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: rgba(94,234,212,0.18) !important; }
.card-title {
  font-size: 0.66rem; font-weight: 600; color: var(--text-muted);
  letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.55rem;
  display: flex; justify-content: space-between; align-items: center;
}
.card-title .hint { color: var(--text-dim); font-weight: 500; letter-spacing: 0.02em; text-transform: none; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; }

/* Streamlit's column gap to match design --gap */
[data-testid="stHorizontalBlock"] { gap: var(--gap) !important; }

/* === Holdings table === */
.holdings { width: 100%; border-collapse: collapse; }
.holdings thead th {
  background: linear-gradient(180deg, var(--surface-2), var(--surface-3));
  color: var(--text-muted);
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  font-size: 0.64rem;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  text-align: right;
  padding: var(--row-h-pad);
  border-bottom: 1px solid var(--border);
}
.holdings thead th:first-child { text-align: left; border-top-left-radius: 6px; }
.holdings thead th:last-child  { border-top-right-radius: 6px; }
.holdings tbody td {
  padding: var(--row-pad);
  text-align: right;
  border-bottom: 1px solid var(--border-soft);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.81rem;
  font-feature-settings: "tnum";
  color: var(--text);
  position: relative;
  letter-spacing: 0.01em;
}
.holdings tbody tr:last-child td { border-bottom: none; }
.holdings tbody td:first-child {
  text-align: left; font-weight: 600; color: var(--accent); letter-spacing: 0.02em;
  text-shadow: 0 0 12px rgba(94,234,212,0.25);
}
.holdings tbody tr { transition: background .15s ease; }
.holdings tbody tr:hover {
  background: linear-gradient(90deg, rgba(94,234,212,0.05), transparent 60%);
}
.holdings .bar-cell { position: relative; }
.holdings .bar-cell .bar {
  position: absolute; left: 0; top: 0; bottom: 0;
  background: linear-gradient(90deg, rgba(94,234,212,0.15), rgba(94,234,212,0.05));
  pointer-events: none;
  border-right: 1px solid rgba(94,234,212,0.18);
}
.holdings .bar-cell.risk .bar {
  background: linear-gradient(90deg, rgba(240,185,11,0.13), rgba(240,185,11,0.04));
  border-right: 1px solid rgba(240,185,11,0.2);
}
.holdings .bar-cell .v { position: relative; padding: 0 0.9rem 0 0; }
.holdings td.up { color: var(--up); }
.holdings td.down { color: var(--down); }

/* === Empty state === */
.empty {
  text-align: center;
  padding: 5rem 2rem;
  color: var(--text-muted);
}
.empty .glyph { font-size: 2.5rem; color: var(--accent); margin-bottom: 1rem; text-shadow: 0 0 24px var(--accent-glow); }
.empty .h { font-size: 1.1rem; font-weight: 600; color: var(--text); margin-bottom: 0.4rem; }
.empty .p { font-size: 0.9rem; max-width: 440px; margin: 0 auto; line-height: 1.55; }
"""


def _density_vars(density: str) -> str:
    if density == "compact":
        return """
          --pad-card: 0.7rem 0.95rem;
          --pad-kpi: 0.7rem 0.95rem;
          --kpi-value: 1.4rem;
          --gap: 0.55rem;
          --row-pad: 0.35rem 0.7rem;
          --row-h-pad: 0.5rem 0.7rem;
          --base-font: 12.5px;
        """
    return """
          --pad-card: 1.15rem 1.35rem;
          --pad-kpi: 1.05rem 1.25rem;
          --kpi-value: 1.85rem;
          --gap: 0.95rem;
          --row-pad: 0.6rem 1rem;
          --row-h-pad: 0.75rem 1rem;
          --base-font: 14px;
        """


def inject_css(density: str = "comfortable"):
    root = f"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
      --bg: {COLORS['bg']};
      --bg-2: {COLORS['bg_2']};
      --surface: {COLORS['surface']};
      --surface-2: {COLORS['surface_2']};
      --surface-3: {COLORS['surface_3']};
      --border: {COLORS['border']};
      --border-soft: {COLORS['border_soft']};
      --text: {COLORS['text']};
      --text-muted: {COLORS['text_muted']};
      --text-dim: {COLORS['text_dim']};
      --up: {COLORS['up']};
      --down: {COLORS['down']};
      --warn: {COLORS['warn']};
      --accent: {COLORS['accent']};
      --accent-rgb: {COLORS['accent_rgb']};
      --accent-dim: {COLORS['accent_dim']};
      --accent-glow: {COLORS['accent_glow']};

      --shadow-card: 0 1px 0 rgba(255,255,255,0.025) inset, 0 1px 2px rgba(0,0,0,0.3), 0 14px 40px -18px rgba(0,0,0,0.55);
      --shadow-soft: 0 1px 0 rgba(255,255,255,0.02) inset, 0 6px 24px -12px rgba(0,0,0,0.5);
      --shadow-glow: 0 0 0 1px rgba(94,234,212,0.18), 0 0 32px -6px rgba(94,234,212,0.25);

      {_density_vars(density)}
    }}
    """
    st.markdown(f"<style>{root}{_STATIC_CSS}</style>", unsafe_allow_html=True)


def chart_layout(height: int = 320, **overrides) -> dict:
    base = dict(
        height=height,
        margin=dict(l=36, r=14, t=12, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text_muted"], size=11),
        xaxis=dict(
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            linecolor=COLORS["border"],
            tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_muted"]),
            showgrid=False,
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            linecolor=COLORS["border"],
            tickfont=dict(family="JetBrains Mono", size=10, color=COLORS["text_muted"]),
            showgrid=True,
            gridwidth=0.5,
        ),
        hoverlabel=dict(
            bgcolor=COLORS["surface_2"],
            bordercolor=COLORS["border"],
            font=dict(family="JetBrains Mono", color=COLORS["text"], size=11),
        ),
        legend=dict(
            font=dict(color=COLORS["text_muted"], size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )
    base.update(overrides)
    return base


def render_topbar(status: str, tickers: list[str] | None = None):
    pill_class = "ok" if status == "active" else "idle"
    pill_text = "Solver Active" if status == "active" else "Idle"
    tickers_html = ""
    if tickers:
        joined = ", ".join(tickers)
        tickers_html = f'<div class="tickers-line">{html.escape(joined)}</div>'
    st.markdown(f"""
    <div class="topbar">
      <div class="brand">
        <span class="mark">◆</span>AlphaAlloc<span class="sub">Portfolio Engine</span>
      </div>
      <div class="topbar-right">
        {tickers_html}
        <span class="pill {pill_class}"><span class="dot"></span>{pill_text}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str | None = None, sub_dir: str = ""):
    sub_html = f'<div class="kpi-sub {sub_dir}">{sub}</div>' if sub is not None else ""
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value tabular">{value}</div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            '<div class="side-title"><span class="mark">◆</span>Configuration</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-section">Universe</div>', unsafe_allow_html=True)
        tickers_raw = st.text_area(
            "Tickers",
            "AAPL,MSFT,GOOGL,AMZN,META,JPM,GS,XOM,CVX,PG,KO,JNJ,UNH,NVDA,TSLA",
            height=80,
            label_visibility="collapsed",
        )
        st.markdown('<div class="side-section">Lookback</div>', unsafe_allow_html=True)
        lookback = st.slider("Lookback (years)", 1, 10, 5, label_visibility="collapsed")

        st.markdown('<div class="side-section">Objective</div>', unsafe_allow_html=True)
        objective = st.selectbox(
            "Objective",
            ["mean_variance", "min_variance"],
            format_func=lambda x: {"mean_variance": "Mean-Variance", "min_variance": "Min Variance"}[x],
            label_visibility="collapsed",
        )
        st.markdown('<div class="side-section">Risk aversion (λ)</div>', unsafe_allow_html=True)
        risk_aversion = st.slider("λ", 0.5, 20.0, 5.0, 0.5, label_visibility="collapsed")

        st.markdown('<div class="side-section">Constraints</div>', unsafe_allow_html=True)
        max_weight = st.slider("Max weight per asset", 0.05, 1.0, 0.20, 0.05)

        st.markdown('<div class="side-section">Forecasting</div>', unsafe_allow_html=True)
        use_ml = st.toggle("ML expected returns", value=True)
        ml_shrinkage = st.slider(
            "ML shrinkage", 0.0, 1.0, 0.5, 0.05,
            help="0 = trust ML; 1 = ignore ML, use historical mean",
        )

        st.markdown('<div class="side-section">Display</div>', unsafe_allow_html=True)
        density = st.segmented_control(
            "Density",
            ["Comfort", "Compact"],
            default=st.session_state.get("density_label", "Comfort"),
            label_visibility="collapsed",
        )
        st.session_state["density_label"] = density or "Comfort"

        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
        run = st.button("Optimize Portfolio", use_container_width=True)

    return dict(
        tickers=[t.strip().upper() for t in tickers_raw.replace("\n", ",").split(",") if t.strip()],
        lookback_years=lookback,
        objective=objective,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        use_ml=use_ml,
        ml_shrinkage=ml_shrinkage,
        density="compact" if (density or "Comfort").lower() == "compact" else "comfortable",
        run=run,
    )


def render_kpis(result: dict):
    er = result["expected_return"] * 100
    vol = result["volatility"] * 100
    sharpe = result["sharpe"]
    weights = np.array(result["weights"])
    concentration = float((weights ** 2).sum())
    n_pos = int((weights > 0.01).sum())

    sign_er = "+" if er >= 0 else ""
    er_dir = "up" if er >= 0 else "down"

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Expected Return</div>
        <div class="kpi-value tabular">{sign_er}{er:.2f}%</div>
        <div class="kpi-sub {er_dir}">Annualized</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Volatility</div>
        <div class="kpi-value tabular">{vol:.2f}%</div>
        <div class="kpi-sub">Annualized σ</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value tabular">{sharpe:.2f}</div>
        <div class="kpi-sub">Risk-adj. return</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Concentration</div>
        <div class="kpi-value tabular">{concentration:.3f}</div>
        <div class="kpi-sub">{n_pos} positions · Herfindahl</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_equity_curve(result: dict, returns: pd.DataFrame) -> go.Figure:
    weights = np.array(result["weights"])
    port_rets = returns @ weights
    cum = (1 + port_rets).cumprod()
    eq_w = np.ones(len(weights)) / len(weights)
    bench = (1 + returns @ eq_w).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bench.index, y=bench.values,
        mode="lines",
        line=dict(color=COLORS["text_dim"], width=1.2, dash="dot"),
        name="Equal-weight",
        hovertemplate="<b>Equal-weight</b><br>%{x|%b %d, %Y}<br>%{y:.3f}x<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values,
        mode="lines",
        line=dict(color=COLORS["accent"], width=2.4, shape="spline", smoothing=0.5),
        fill="tozeroy",
        fillcolor=f"rgba({COLORS['accent_rgb']}, 0.10)",
        name="Optimized",
        hovertemplate="<b>Optimized</b><br>%{x|%b %d, %Y}<br>%{y:.3f}x<extra></extra>",
    ))

    last_val = float(cum.iloc[-1])
    fig.add_annotation(
        x=cum.index[-1], y=last_val,
        text=f"  {last_val:.2f}x",
        showarrow=False,
        font=dict(family="JetBrains Mono", color=COLORS["accent"], size=12),
        xanchor="left", yanchor="middle",
    )

    layout = chart_layout(height=320, margin=dict(l=42, r=54, t=14, b=28))
    layout["yaxis"]["tickformat"] = ".2f"
    layout["yaxis"]["ticksuffix"] = "x"
    y_min = min(0.85, float(min(cum.min(), bench.min())) * 0.95)
    y_max = float(max(cum.max(), bench.max())) * 1.05
    layout["yaxis"]["range"] = [y_min, y_max]
    layout["hovermode"] = "x unified"
    layout["xaxis"]["tickformat"] = "%b %Y"
    layout["xaxis"]["type"] = "date"
    fig.update_layout(**layout)
    return fig


def render_allocation_donut(result: dict) -> go.Figure:
    df = pd.DataFrame({"ticker": result["tickers"], "weight": result["weights"]})
    df = df[df["weight"] > 1e-3].sort_values("weight", ascending=False)

    n = len(df)
    palette = [
        f"rgba({COLORS['accent_rgb']}, {0.95 - i * 0.55 / max(n - 1, 1):.3f})"
        for i in range(n)
    ]

    fig = go.Figure(go.Pie(
        labels=df["ticker"],
        values=df["weight"],
        hole=0.62,
        marker=dict(colors=palette, line=dict(color=COLORS["surface"], width=2)),
        textfont=dict(family="JetBrains Mono", color=COLORS["text"], size=11),
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="<b>%{label}</b><br>%{percent}<br>w = %{value:.3f}<extra></extra>",
        sort=False,
        direction="clockwise",
    ))

    fig.add_annotation(
        text=f"<b>{n}</b><br><span style='font-size:10px; color:{COLORS['text_muted']};'>POSITIONS</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(family="Inter", size=24, color=COLORS["text"]),
    )
    fig.update_layout(**chart_layout(height=320, margin=dict(l=30, r=30, t=14, b=14)))
    return fig


def render_risk_contribution(result: dict, Sigma: np.ndarray) -> go.Figure:
    w = np.array(result["weights"])
    tickers = result["tickers"]
    port_var = float(w @ Sigma @ w)
    if port_var <= 0:
        return go.Figure()
    mrc = Sigma @ w
    rc = w * mrc / np.sqrt(port_var)
    df = pd.DataFrame({"ticker": tickers, "rc": rc})
    df = df[df["rc"].abs() > 1e-6].sort_values("rc", ascending=True)

    fig = go.Figure(go.Bar(
        y=df["ticker"], x=df["rc"],
        orientation="h",
        marker=dict(
            color=[COLORS["accent"] if v >= 0 else COLORS["down"] for v in df["rc"]],
            line=dict(width=0),
        ),
        text=[f"{v * 100:.2f}%" for v in df["rc"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", color=COLORS["text_muted"], size=10),
        hovertemplate="<b>%{y}</b><br>RC: %{x:.4f}<extra></extra>",
    ))
    layout = chart_layout(height=300, margin=dict(l=48, r=48, t=10, b=24))
    layout["xaxis"]["tickformat"] = ".1%"
    layout["yaxis"]["tickfont"] = dict(family="JetBrains Mono", size=10, color=COLORS["text"])
    layout["yaxis"]["automargin"] = True
    fig.update_layout(**layout)
    return fig


def render_correlation(returns: pd.DataFrame, tickers: list[str]) -> go.Figure:
    corr = returns[tickers].corr().values
    fig = go.Figure(go.Heatmap(
        z=corr,
        x=tickers, y=tickers,
        colorscale=[
            [0.0, COLORS["down"]],
            [0.5, COLORS["surface_2"]],
            [1.0, COLORS["accent"]],
        ],
        zmin=-1, zmax=1,
        showscale=True,
        colorbar=dict(
            thickness=8, len=0.7,
            tickfont=dict(family="JetBrains Mono", color=COLORS["text_muted"], size=9),
            outlinewidth=0,
        ),
        hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>ρ = %{z:.2f}<extra></extra>",
    ))
    layout = chart_layout(height=320, margin=dict(l=50, r=30, t=10, b=40))
    layout["xaxis"]["tickfont"] = dict(family="JetBrains Mono", size=9, color=COLORS["text_muted"])
    layout["yaxis"]["tickfont"] = dict(family="JetBrains Mono", size=9, color=COLORS["text_muted"])
    layout["xaxis"]["showgrid"] = False
    layout["yaxis"]["showgrid"] = False
    layout["xaxis"]["tickangle"] = -45
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)
    return fig


def render_efficient_frontier(mu: np.ndarray, Sigma: np.ndarray, current: dict) -> go.Figure:
    import cvxpy as cp
    n = len(mu)
    pts = []
    target_rets = np.linspace(mu.min(), mu.max(), 25)
    for tr in target_rets:
        w = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))),
            [cp.sum(w) == 1, w >= 0, mu @ w >= tr],
        )
        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status == "optimal":
                wv = np.array(w.value).flatten()
                pts.append((float(np.sqrt(wv @ Sigma @ wv)), float(mu @ wv)))
        except Exception:
            pass

    fig = go.Figure()
    rng = np.random.default_rng(0)
    rand = rng.dirichlet(np.ones(n), 800)
    rand_sig = np.sqrt(np.einsum("ij,jk,ik->i", rand, Sigma, rand))
    rand_mu = rand @ mu
    fig.add_trace(go.Scatter(
        x=rand_sig, y=rand_mu, mode="markers",
        marker=dict(color=COLORS["text_dim"], size=3, opacity=0.4),
        hoverinfo="skip",
    ))

    if pts:
        xs, ys = zip(*pts)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=COLORS["accent"], width=2, shape="spline", smoothing=0.6),
            hovertemplate="σ = %{x:.3f}<br>μ = %{y:.3f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[current["volatility"]], y=[current["expected_return"]],
        mode="markers",
        marker=dict(color=COLORS["accent"], size=14, symbol="diamond",
                    line=dict(color=COLORS["bg"], width=2)),
        hovertemplate="<b>Current</b><br>σ = %{x:.3f}<br>μ = %{y:.3f}<extra></extra>",
    ))

    layout = chart_layout(height=300, margin=dict(l=50, r=14, t=10, b=38))
    layout["xaxis"]["tickformat"] = ".0%"
    layout["yaxis"]["tickformat"] = ".0%"
    layout["xaxis"]["title"] = dict(text="Volatility (σ)", font=dict(size=10, color=COLORS["text_muted"]))
    layout["yaxis"]["title"] = dict(text="Expected Return (μ)", font=dict(size=10, color=COLORS["text_muted"]))
    fig.update_layout(**layout)
    return fig


def render_holdings_table(result: dict, mu: np.ndarray, Sigma: np.ndarray):
    """Hand-built HTML table with inline weight + risk% bars (matches AlphaAlloc design)."""
    w = np.array(result["weights"])
    tickers = result["tickers"]
    sigmas = np.sqrt(np.diag(Sigma))
    port_var = float(w @ Sigma @ w)
    rc = w * (Sigma @ w) / np.sqrt(port_var) if port_var > 0 else np.zeros_like(w)
    rc_pos_sum = float(np.maximum(rc, 0).sum())
    rc_pct = rc / rc_pos_sum if rc_pos_sum > 0 else np.zeros_like(rc)

    rows = [
        {"t": tickers[i], "w": float(w[i]), "mu": float(mu[i]),
         "sig": float(sigmas[i]), "rc": float(rc[i]), "rc_pct": float(rc_pct[i])}
        for i in range(len(tickers)) if w[i] > 1e-4
    ]
    rows.sort(key=lambda r: r["w"], reverse=True)
    if not rows:
        st.markdown("<div style='color:var(--text-muted); padding:1rem;'>No active positions.</div>",
                    unsafe_allow_html=True)
        return

    max_w = max(r["w"] for r in rows)
    max_rc = max(r["rc_pct"] for r in rows) if any(r["rc_pct"] > 0 for r in rows) else 1.0

    body = []
    for r in rows:
        mu_class = "up" if r["mu"] >= 0 else "down"
        mu_sign = "+" if r["mu"] >= 0 else ""
        w_pct = r["w"] / max_w * 100
        rc_pct = (r["rc_pct"] / max_rc * 100) if max_rc > 0 else 0
        body.append(f"""
        <tr>
          <td>{html.escape(r['t'])}</td>
          <td class="bar-cell"><div class="bar" style="width:{w_pct:.1f}%"></div><span class="v">{r['w']*100:.2f}%</span></td>
          <td class="{mu_class}">{mu_sign}{r['mu']*100:.2f}%</td>
          <td>{r['sig']*100:.2f}%</td>
          <td>{r['rc']:.4f}</td>
          <td class="bar-cell risk"><div class="bar" style="width:{rc_pct:.1f}%"></div><span class="v">{r['rc_pct']*100:.1f}%</span></td>
        </tr>""")

    table = f"""
    <table class="holdings">
      <thead>
        <tr>
          <th>Ticker</th><th>Weight</th><th>μ (ann.)</th><th>σ (ann.)</th><th>Risk Contrib.</th><th>Risk %</th>
        </tr>
      </thead>
      <tbody>{''.join(body)}</tbody>
    </table>
    """
    st.markdown(table, unsafe_allow_html=True)


def render_empty_state():
    st.markdown("""
    <div class="empty">
      <div class="glyph">◆</div>
      <div class="h">Configure portfolio universe</div>
      <div class="p">
        Set tickers, objective, and constraints in the sidebar, then run the optimizer to see allocations,
        risk decomposition, and the efficient frontier.
      </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    # First-pass CSS so the sidebar renders correctly even before density is known.
    pre_density = st.session_state.get("density_label", "Comfort").lower()
    pre_density = "compact" if pre_density == "compact" else "comfortable"
    inject_css(density=pre_density)

    cfg = render_sidebar()

    # Re-inject if the density toggle changed this run.
    if cfg["density"] != pre_density:
        inject_css(density=cfg["density"])

    if not cfg["run"] and "result" not in st.session_state:
        render_topbar(status="idle", tickers=cfg["tickers"])
        render_empty_state()
        return

    if cfg["run"]:
        with st.spinner("Solving optimization..."):
            try:
                req = AllocationRequest(
                    tickers=cfg["tickers"],
                    lookback_years=cfg["lookback_years"],
                    objective=cfg["objective"],
                    risk_aversion=cfg["risk_aversion"],
                    max_weight=cfg["max_weight"],
                    use_ml=cfg["use_ml"],
                    ml_shrinkage=cfg["ml_shrinkage"],
                )
                result = run_allocation(req)
                st.session_state["result"]  = result
                st.session_state["returns"] = result["returns"]
                st.session_state["Sigma"]   = result["Sigma"]
                st.session_state["mu"]      = result["mu"]
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                return

    result  = st.session_state["result"]
    returns = st.session_state["returns"]
    Sigma   = st.session_state["Sigma"]
    mu      = st.session_state["mu"]

    render_topbar(status="active", tickers=result["tickers"])
    render_kpis(result)

    # Row 1: Equity curve (2) | Donut (1)
    c1, c2 = st.columns([2, 1], gap="small")
    with c1:
        with st.container(border=True):
            date_idx = returns.index
            date_hint = f"{date_idx[0].date().isoformat()} → {date_idx[-1].date().isoformat()}"
            st.markdown(
                f'<div class="card-title"><span>Cumulative Performance — Optimized vs Equal-Weight</span><span class="hint">{date_hint}</span></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(render_equity_curve(result, returns), use_container_width=True, config={"displayModeBar": False})
    with c2:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title"><span>Allocation</span><span class="hint">long-only</span></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(render_allocation_donut(result), use_container_width=True, config={"displayModeBar": False})

    # Row 2: Risk Contribution | Correlation | Frontier
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title"><span>Risk Contribution</span><span class="hint">marginal × weight</span></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(render_risk_contribution(result, Sigma), use_container_width=True, config={"displayModeBar": False})
    with c2:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title"><span>Correlation Matrix</span><span class="hint">ρ ∈ [−1, 1]</span></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(render_correlation(returns, result["tickers"]), use_container_width=True, config={"displayModeBar": False})
    with c3:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title"><span>Efficient Frontier</span><span class="hint">μ vs σ</span></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(render_efficient_frontier(mu, Sigma, result), use_container_width=True, config={"displayModeBar": False})

    # Row 3: Holdings
    n_active = int(np.sum(np.array(result["weights"]) > 1e-4))
    with st.container(border=True):
        st.markdown(
            f'<div class="card-title"><span>Holdings</span><span class="hint">sorted by weight · {n_active} active</span></div>',
            unsafe_allow_html=True,
        )
        render_holdings_table(result, mu, Sigma)


if __name__ == "__main__":
    main()
