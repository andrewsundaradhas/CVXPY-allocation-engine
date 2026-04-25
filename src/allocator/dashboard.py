from __future__ import annotations

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

# === Design tokens ===
COLORS = {
    "bg":         "#0B0E13",
    "surface":    "#11151C",
    "surface_2":  "#171C26",
    "border":     "#1F2630",
    "text":       "#E6E9EF",
    "text_muted": "#8A93A6",
    "text_dim":   "#5A6273",
    "up":         "#16C784",
    "down":       "#EA3943",
    "warn":       "#F0B90B",
    "accent":     "#5EEAD4",
    "accent_dim": "rgba(45, 212, 191, 0.13)",
}

FONTS = {
    "ui":  "'Inter', -apple-system, system-ui, sans-serif",
    "num": "'JetBrains Mono', 'SF Mono', Consolas, monospace",
}


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {{
        background: {COLORS['bg']};
        color: {COLORS['text']};
        font-family: {FONTS['ui']};
    }}

    #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }}

    section[data-testid="stSidebar"] {{
        background: {COLORS['surface']};
        border-right: 1px solid {COLORS['border']};
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {{
        color: {COLORS['text_muted']};
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}

    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background: {COLORS['surface_2']};
        border: 1px solid {COLORS['border']};
        color: {COLORS['text']};
        font-family: {FONTS['num']};
        font-size: 0.85rem;
    }}

    .stSlider [data-baseweb="slider"] > div > div > div {{
        background: {COLORS['accent']};
    }}

    .stButton > button {{
        background: {COLORS['accent']};
        color: {COLORS['bg']};
        border: none;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.03em;
        padding: 0.55rem 1.2rem;
        border-radius: 6px;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        background: {COLORS['text']};
        transform: translateY(-1px);
    }}

    .card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }}
    .card-title {{
        font-size: 0.7rem;
        font-weight: 600;
        color: {COLORS['text_muted']};
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }}

    .kpi {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        position: relative;
        overflow: hidden;
    }}
    .kpi::before {{
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: {COLORS['accent']};
        opacity: 0.7;
    }}
    .kpi-label {{
        font-size: 0.7rem;
        font-weight: 600;
        color: {COLORS['text_muted']};
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}
    .kpi-value {{
        font-family: {FONTS['num']};
        font-size: 1.7rem;
        font-weight: 600;
        color: {COLORS['text']};
        line-height: 1.2;
        margin: 0.3rem 0 0.2rem;
        font-feature-settings: "tnum";
    }}
    .kpi-delta {{
        font-family: {FONTS['num']};
        font-size: 0.8rem;
        font-weight: 500;
    }}
    .kpi-delta.up   {{ color: {COLORS['up']}; }}
    .kpi-delta.down {{ color: {COLORS['down']}; }}

    .pill {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    .pill.ok    {{ background: {COLORS['up']}22;   color: {COLORS['up']}; }}
    .pill.idle  {{ background: {COLORS['text_dim']}22; color: {COLORS['text_muted']}; }}

    .tabular {{ font-family: {FONTS['num']}; font-feature-settings: "tnum"; }}

    .topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0 1rem;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 1.2rem;
    }}
    .brand {{
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.02em;
    }}
    .brand .mark {{ color: {COLORS['accent']}; margin-right: 6px; }}
    </style>
    """, unsafe_allow_html=True)


inject_css()


def chart_layout(height: int = 320, **overrides) -> dict:
    base = dict(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
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


def render_topbar(status: str, n_assets: int):
    pill_class = "ok" if status == "active" else "idle"
    pill_text = "Solver Active" if status == "active" else "Idle"
    st.markdown(f"""
    <div class="topbar">
      <div class="brand"><span class="mark">◆</span>AlphaAlloc <span style="color:{COLORS['text_dim']}; font-weight:400; margin-left:8px;">Portfolio Engine</span></div>
      <div>
        <span style="color:{COLORS['text_muted']}; font-size:0.78rem; margin-right:14px;">{n_assets} assets</span>
        <span class="pill {pill_class}">● {pill_text}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str | None = None, delta_dir: str = "up"):
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta_dir == "up" else "▼"
        delta_html = f'<div class="kpi-delta {delta_dir}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value tabular">{value}</div>
      {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            f"<div style='padding:0.5rem 0 1rem; font-weight:700; color:{COLORS['text']}; font-size:1rem;'>◆ Configuration</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Universe**")
        tickers_raw = st.text_area(
            "Tickers",
            "AAPL,MSFT,GOOGL,AMZN,META,JPM,GS,XOM,CVX,PG,KO,JNJ,UNH,NVDA,TSLA",
            height=80,
            label_visibility="collapsed",
        )
        lookback = st.slider("Lookback (years)", 1, 10, 5)

        st.markdown("**Objective**")
        objective = st.selectbox(
            "Objective",
            ["mean_variance", "min_variance"],
            format_func=lambda x: {"mean_variance": "Mean-Variance", "min_variance": "Min Variance"}[x],
            label_visibility="collapsed",
        )
        risk_aversion = st.slider("Risk aversion (λ)", 0.5, 20.0, 5.0, 0.5)

        st.markdown("**Constraints**")
        max_weight = st.slider("Max weight per asset", 0.05, 1.0, 0.20, 0.05)

        st.markdown("**Forecasting**")
        use_ml = st.toggle("ML expected returns", value=True)
        ml_shrinkage = st.slider(
            "ML shrinkage", 0.0, 1.0, 0.5, 0.05,
            help="0 = trust ML; 1 = ignore ML, use historical mean",
        )

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        run = st.button("⚡ Optimize Portfolio", use_container_width=True)

    return dict(
        tickers=[t.strip().upper() for t in tickers_raw.replace("\n", ",").split(",") if t.strip()],
        lookback_years=lookback,
        objective=objective,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        use_ml=use_ml,
        ml_shrinkage=ml_shrinkage,
        run=run,
    )


def render_kpis(result: dict, prev_result: dict | None = None):
    cols = st.columns(4, gap="small")

    er = result["expected_return"] * 100
    vol = result["volatility"] * 100
    sharpe = result["sharpe"]
    weights = np.array(result["weights"])
    concentration = float((weights ** 2).sum())  # Herfindahl

    items = [
        ("Expected Return", f"{er:+.2f}%", "Annualized"),
        ("Volatility",      f"{vol:.2f}%", "Annualized σ"),
        ("Sharpe Ratio",    f"{sharpe:.2f}", "Risk-adj. return"),
        ("Concentration",   f"{concentration:.3f}", f"{int((weights > 0.01).sum())} positions"),
    ]
    for col, (label, value, sub) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="kpi">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value tabular">{value}</div>
              <div class="kpi-delta" style="color:{COLORS['text_dim']};">{sub}</div>
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
        line=dict(color=COLORS["accent"], width=2.2),
        fill="tozeroy",
        fillcolor=COLORS["accent_dim"],
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

    layout = chart_layout(height=340)
    layout["yaxis"]["tickformat"] = ".2f"
    layout["yaxis"]["ticksuffix"] = "x"
    fig.update_layout(**layout)
    return fig


def render_allocation_donut(result: dict) -> go.Figure:
    df = pd.DataFrame({"ticker": result["tickers"], "weight": result["weights"]})
    df = df[df["weight"] > 1e-3].sort_values("weight", ascending=False)

    n = len(df)
    palette = [
        f"rgba(94, 234, 212, {0.95 - i * 0.55 / max(n - 1, 1)})"
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
    ))

    fig.add_annotation(
        text=f"<b>{n}</b><br><span style='font-size:10px; color:{COLORS['text_muted']};'>POSITIONS</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(family="Inter", size=22, color=COLORS["text"]),
    )
    fig.update_layout(**chart_layout(height=340))
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
        hovertemplate="<b>%{y}</b><br>Risk contribution: %{x:.4f}<extra></extra>",
    ))
    layout = chart_layout(height=320)
    layout["xaxis"]["tickformat"] = ".2%"
    layout["yaxis"]["tickfont"] = dict(family="JetBrains Mono", size=10, color=COLORS["text"])
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
            thickness=10, len=0.7,
            tickfont=dict(family="JetBrains Mono", color=COLORS["text_muted"], size=9),
            outlinewidth=0,
        ),
        hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>ρ = %{z:.2f}<extra></extra>",
    ))
    layout = chart_layout(height=340)
    layout["xaxis"]["tickfont"] = dict(family="JetBrains Mono", size=9, color=COLORS["text_muted"])
    layout["yaxis"]["tickfont"] = dict(family="JetBrains Mono", size=9, color=COLORS["text_muted"])
    layout["xaxis"]["showgrid"] = False
    layout["yaxis"]["showgrid"] = False
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

    if not pts:
        return go.Figure()
    xs, ys = zip(*pts)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=COLORS["accent"], width=2),
        hovertemplate="σ = %{x:.3f}<br>μ = %{y:.3f}<extra></extra>",
    ))
    rng = np.random.default_rng(0)
    rand = rng.dirichlet(np.ones(n), 800)
    rand_sig = np.sqrt(np.einsum("ij,jk,ik->i", rand, Sigma, rand))
    rand_mu = rand @ mu
    fig.add_trace(go.Scatter(
        x=rand_sig, y=rand_mu, mode="markers",
        marker=dict(color=COLORS["text_dim"], size=3, opacity=0.4),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[current["volatility"]], y=[current["expected_return"]],
        mode="markers",
        marker=dict(color=COLORS["accent"], size=14, symbol="diamond",
                    line=dict(color=COLORS["bg"], width=2)),
        hovertemplate="<b>Current</b><br>σ = %{x:.3f}<br>μ = %{y:.3f}<extra></extra>",
    ))

    layout = chart_layout(height=320)
    layout["xaxis"]["title"] = dict(text="Volatility (σ)", font=dict(size=10, color=COLORS["text_muted"]))
    layout["yaxis"]["title"] = dict(text="Expected Return (μ)", font=dict(size=10, color=COLORS["text_muted"]))
    layout["xaxis"]["tickformat"] = ".1%"
    layout["yaxis"]["tickformat"] = ".1%"
    fig.update_layout(**layout)
    return fig


def render_holdings_table(result: dict, mu: np.ndarray, Sigma: np.ndarray):
    w = np.array(result["weights"])
    tickers = result["tickers"]
    sigmas = np.sqrt(np.diag(Sigma))
    port_var = float(w @ Sigma @ w)
    rc = w * (Sigma @ w) / np.sqrt(port_var) if port_var > 0 else np.zeros_like(w)

    df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": w,
        "μ (ann.)": mu,
        "σ (ann.)": sigmas,
        "Risk Contrib.": rc,
        "Risk %": rc / rc.sum() if rc.sum() > 0 else rc,
    })
    df = df[df["Weight"] > 1e-4].sort_values("Weight", ascending=False).reset_index(drop=True)

    styled = (
        df.style
        .format({
            "Weight": "{:.2%}",
            "μ (ann.)": "{:+.2%}",
            "σ (ann.)": "{:.2%}",
            "Risk Contrib.": "{:.4f}",
            "Risk %": "{:.1%}",
        })
        .background_gradient(subset=["Weight"], cmap="Greens", vmin=0, vmax=df["Weight"].max())
        .background_gradient(subset=["Risk %"], cmap="Oranges", vmin=0, vmax=df["Risk %"].max())
        .set_properties(**{
            "background-color": COLORS["surface"],
            "color": COLORS["text"],
            "font-family": "JetBrains Mono, monospace",
            "font-size": "0.82rem",
            "border-color": COLORS["border"],
        })
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", COLORS["surface_2"]),
                ("color", COLORS["text_muted"]),
                ("font-family", "Inter, sans-serif"),
                ("font-weight", "600"),
                ("font-size", "0.7rem"),
                ("letter-spacing", "0.06em"),
                ("text-transform", "uppercase"),
                ("text-align", "right"),
                ("padding", "0.7rem 0.9rem"),
                ("border-bottom", f"1px solid {COLORS['border']}"),
            ]},
            {"selector": "tbody td", "props": [
                ("padding", "0.55rem 0.9rem"),
                ("text-align", "right"),
                ("border-bottom", f"1px solid {COLORS['border']}"),
            ]},
            {"selector": "tbody td:first-child", "props": [
                ("text-align", "left"),
                ("font-weight", "600"),
                ("color", COLORS["accent"]),
            ]},
        ])
        .hide(axis="index")
    )
    st.markdown(styled.to_html(), unsafe_allow_html=True)


def render_empty_state():
    st.markdown(f"""
    <div style="text-align:center; padding:5rem 2rem; color:{COLORS['text_muted']};">
      <div style="font-size:2.5rem; margin-bottom:1rem;">◆</div>
      <div style="font-size:1.1rem; font-weight:600; color:{COLORS['text']}; margin-bottom:0.4rem;">
        Configure portfolio universe
      </div>
      <div style="font-size:0.9rem; max-width:420px; margin:0 auto;">
        Set tickers, objective, and constraints in the sidebar, then run the optimizer to see allocations,
        risk decomposition, and the efficient frontier.
      </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    cfg = render_sidebar()

    if not cfg["run"] and "result" not in st.session_state:
        render_topbar(status="idle", n_assets=len(cfg["tickers"]))
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

    render_topbar(status="active", n_assets=len(result["tickers"]))
    render_kpis(result)

    c1, c2 = st.columns([2, 1], gap="small")
    with c1:
        st.markdown('<div class="card"><div class="card-title">Cumulative Performance — Optimized vs Equal-Weight</div>', unsafe_allow_html=True)
        st.plotly_chart(render_equity_curve(result, returns), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Allocation</div>', unsafe_allow_html=True)
        st.plotly_chart(render_allocation_donut(result), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown('<div class="card"><div class="card-title">Risk Contribution</div>', unsafe_allow_html=True)
        st.plotly_chart(render_risk_contribution(result, Sigma), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Correlation Matrix</div>', unsafe_allow_html=True)
        st.plotly_chart(render_correlation(returns, result["tickers"]), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card"><div class="card-title">Efficient Frontier</div>', unsafe_allow_html=True)
        st.plotly_chart(render_efficient_frontier(mu, Sigma, result), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">Holdings</div>', unsafe_allow_html=True)
    render_holdings_table(result, mu, Sigma)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
