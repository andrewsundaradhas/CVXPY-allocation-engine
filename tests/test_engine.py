from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from allocator.engine import AllocationRequest, run_allocation


def _synthetic_prices(tickers: list[str], n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    daily = rng.standard_normal((n, len(tickers))) * 0.012
    prices = 100.0 * np.exp(np.cumsum(daily, axis=0))
    return pd.DataFrame(prices, index=idx, columns=tickers)


def test_run_allocation_with_ml_path():
    tickers = ["A", "B", "C", "D", "E"]
    prices = _synthetic_prices(tickers, n=600)

    with patch("allocator.engine.load_prices", return_value=prices):
        out = run_allocation(AllocationRequest(
            tickers=tickers,
            objective="mean_variance",
            risk_aversion=5.0,
            max_weight=0.4,
            use_ml=True,
        ))

    assert out["tickers"] == tickers
    assert len(out["weights"]) == len(tickers)
    np.testing.assert_allclose(out["weights"].sum(), 1.0, atol=1e-6)
    assert (out["weights"] >= -1e-9).all()
    assert (out["weights"] <= 0.4 + 1e-6).all()
    assert out["status"] in ("optimal", "optimal_inaccurate")
    assert out["volatility"] > 0
    assert len(out["mu_used"]) == len(tickers)


def test_run_allocation_without_ml_falls_back_to_historical_mean():
    tickers = ["A", "B", "C"]
    prices = _synthetic_prices(tickers, n=200)

    with patch("allocator.engine.load_prices", return_value=prices):
        out = run_allocation(AllocationRequest(
            tickers=tickers,
            objective="min_variance",
            max_weight=0.6,
            use_ml=False,
        ))

    np.testing.assert_allclose(out["weights"].sum(), 1.0, atol=1e-6)
    assert out["status"] in ("optimal", "optimal_inaccurate")


def test_short_history_skips_ml_branch_even_when_requested():
    """use_ml=True but len(rets) <= 300 should fall through to the historical-mean path."""
    tickers = ["A", "B", "C"]
    prices = _synthetic_prices(tickers, n=250)

    with patch("allocator.engine.load_prices", return_value=prices):
        out = run_allocation(AllocationRequest(
            tickers=tickers,
            objective="mean_variance",
            risk_aversion=5.0,
            max_weight=0.5,
            use_ml=True,
        ))

    np.testing.assert_allclose(out["weights"].sum(), 1.0, atol=1e-6)
