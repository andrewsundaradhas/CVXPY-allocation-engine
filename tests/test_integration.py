"""End-to-end integration: real run_allocation output → dashboard render functions.

This locks down the contract between the ML pipeline and the dashboard:
- engine artifacts (returns / Sigma / mu) have the right shapes and units
- every dashboard render function consumes the result without exception
- numeric invariants hold (weights sum to 1, mu/Sigma finite, RC sums to vol)
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from allocator.engine import AllocationRequest, run_allocation


def _synthetic_prices(tickers: list[str], n: int = 700, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    daily = rng.standard_normal((n, len(tickers))) * 0.012
    prices = 100.0 * np.exp(np.cumsum(daily, axis=0))
    return pd.DataFrame(prices, index=idx, columns=tickers)


def _real_result_with_ml(tickers: list[str]) -> dict:
    prices = _synthetic_prices(tickers, n=700)
    with patch("allocator.engine.load_prices", return_value=prices):
        return run_allocation(AllocationRequest(
            tickers=tickers,
            objective="mean_variance",
            risk_aversion=5.0,
            max_weight=0.4,
            use_ml=True,
            ml_shrinkage=0.5,
        ))


def test_engine_artifact_units_and_shapes():
    """Sigma is annualized, mu is annualized, returns are daily."""
    tickers = ["A", "B", "C", "D", "E"]
    out = _real_result_with_ml(tickers)

    assert isinstance(out["returns"], pd.DataFrame)
    assert list(out["returns"].columns) == tickers
    # Daily log returns: typical |r| < 0.1 on synthetic σ_d=1.2%
    assert out["returns"].abs().mean().max() < 0.05

    assert isinstance(out["Sigma"], np.ndarray)
    assert out["Sigma"].shape == (len(tickers), len(tickers))
    np.testing.assert_allclose(out["Sigma"], out["Sigma"].T, atol=1e-12)
    eigs = np.linalg.eigvalsh(out["Sigma"])
    assert eigs.min() > 0  # PSD after Ledoit-Wolf
    # Annualized: diag should be ~ (σ_daily)² * 252 ≈ 0.012² × 252 ≈ 0.036
    diag = np.diag(out["Sigma"])
    assert (diag > 0.005).all() and (diag < 1.0).all()

    assert isinstance(out["mu"], np.ndarray)
    assert out["mu"].shape == (len(tickers),)
    assert np.isfinite(out["mu"]).all()
    # Annualized monthly-blend: stays in a sane band (synthetic, no drift)
    assert (np.abs(out["mu"]) < 2.0).all()


def test_portfolio_invariants_match_engine_summary():
    """expected_return, volatility, sharpe in the summary equal w·μ, √(w'Σw), and the ratio."""
    tickers = ["A", "B", "C", "D"]
    out = _real_result_with_ml(tickers)
    w = np.array(out["weights"])
    mu = out["mu"]
    Sigma = out["Sigma"]

    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-9).all()

    expected = float(mu @ w)
    vol = float(np.sqrt(w @ Sigma @ w))
    np.testing.assert_allclose(out["expected_return"], expected, rtol=1e-6)
    np.testing.assert_allclose(out["volatility"], vol, rtol=1e-6)
    if vol > 0:
        np.testing.assert_allclose(out["sharpe"], expected / vol, rtol=1e-6)


def test_risk_contributions_sum_to_portfolio_volatility():
    """∑(w_i · (Σw)_i / √(w'Σw)) = √(w'Σw). The dashboard chart and table both rely on this."""
    out = _real_result_with_ml(["A", "B", "C", "D", "E"])
    w = np.array(out["weights"])
    Sigma = out["Sigma"]
    port_var = float(w @ Sigma @ w)
    rc = w * (Sigma @ w) / np.sqrt(port_var)
    np.testing.assert_allclose(rc.sum(), np.sqrt(port_var), rtol=1e-9)


def test_every_dashboard_render_function_consumes_real_engine_output():
    """Smoke: each dashboard chart/table builder must run without exception on a real result."""
    from allocator.dashboard import (
        render_allocation_donut,
        render_correlation,
        render_efficient_frontier,
        render_equity_curve,
        render_risk_contribution,
    )

    tickers = ["A", "B", "C", "D", "E"]
    out = _real_result_with_ml(tickers)

    eq = render_equity_curve(out, out["returns"])
    assert isinstance(eq, go.Figure)
    assert len(eq.data) == 2  # benchmark + optimized
    # The optimized line is the second trace; final cumprod value is finite and >0
    final_y = eq.data[1].y[-1]
    assert np.isfinite(final_y) and final_y > 0

    donut = render_allocation_donut(out)
    assert isinstance(donut, go.Figure)
    # Pie slice values must equal the active weights
    active_weights = sorted([w for w in out["weights"] if w > 1e-3], reverse=True)
    np.testing.assert_allclose(sorted(donut.data[0].values, reverse=True), active_weights, atol=1e-9)

    rc_fig = render_risk_contribution(out, out["Sigma"])
    assert isinstance(rc_fig, go.Figure)
    if len(rc_fig.data):
        rc_values = np.array(rc_fig.data[0].x)
        assert np.isfinite(rc_values).all()

    corr_fig = render_correlation(out["returns"], out["tickers"])
    assert isinstance(corr_fig, go.Figure)
    z = np.array(corr_fig.data[0].z)
    np.testing.assert_allclose(np.diag(z), 1.0, atol=1e-9)
    assert (z >= -1.0 - 1e-9).all() and (z <= 1.0 + 1e-9).all()

    ef_fig = render_efficient_frontier(out["mu"], out["Sigma"], out)
    assert isinstance(ef_fig, go.Figure)
    # Dirichlet cloud + (optionally) frontier line + current-portfolio diamond
    assert len(ef_fig.data) >= 2
    diamond = ef_fig.data[-1]
    np.testing.assert_allclose(diamond.x[0], out["volatility"], rtol=1e-9)
    np.testing.assert_allclose(diamond.y[0], out["expected_return"], rtol=1e-9)


def test_holdings_table_html_renders_with_real_values():
    """Hand-built HTML table must include each active ticker, weight pct, and risk pct."""
    from allocator.dashboard import render_holdings_table

    tickers = ["A", "B", "C", "D"]
    out = _real_result_with_ml(tickers)

    captured: list[str] = []
    with patch("allocator.dashboard.st.markdown", side_effect=lambda html, **kw: captured.append(html)):
        render_holdings_table(out, out["mu"], out["Sigma"])

    rendered = "".join(captured)
    assert "<table class=\"holdings\">" in rendered
    for t, w in zip(out["tickers"], out["weights"]):
        if w > 1e-4:
            assert f">{t}</td>" in rendered, f"missing row for {t}"
    # Bar widths should be finite percent values, not NaN
    assert "NaN" not in rendered
    # +/- mu sign rendering
    assert ("class=\"up\"" in rendered) or ("class=\"down\"" in rendered)


def test_no_ml_path_still_provides_dashboard_artifacts():
    """When use_ml=False, returns/Sigma/mu must still be populated for the dashboard."""
    tickers = ["A", "B", "C"]
    prices = _synthetic_prices(tickers, n=200)
    with patch("allocator.engine.load_prices", return_value=prices):
        out = run_allocation(AllocationRequest(
            tickers=tickers,
            objective="min_variance",
            max_weight=0.7,
            use_ml=False,
        ))

    assert out["returns"].shape[1] == len(tickers)
    assert out["Sigma"].shape == (len(tickers), len(tickers))
    assert out["mu"].shape == (len(tickers),)
    np.testing.assert_allclose(np.array(out["weights"]).sum(), 1.0, atol=1e-6)
