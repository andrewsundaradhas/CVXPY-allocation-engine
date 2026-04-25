from __future__ import annotations

import numpy as np
import pandas as pd

from allocator.ml.blend import blend_mu


def _toy_inputs() -> tuple[pd.Series, pd.Series]:
    tickers = ["A", "B", "C", "D"]
    mu_pred = pd.Series([0.02, -0.01, 0.005, -0.015], index=tickers)
    hist = pd.Series([0.008, 0.006, 0.010, 0.004], index=tickers)
    return mu_pred, hist


def test_shrinkage_zero_uses_model_plus_grand_mean():
    mu_pred, hist = _toy_inputs()
    out = blend_mu(mu_pred, hist, shrinkage=0.0)
    expected = mu_pred + hist.mean()
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_shrinkage_one_returns_prior():
    mu_pred, hist = _toy_inputs()
    out = blend_mu(mu_pred, hist, shrinkage=1.0)
    pd.testing.assert_series_equal(out, hist, check_names=False)


def test_shrinkage_half_is_average():
    mu_pred, hist = _toy_inputs()
    out = blend_mu(mu_pred, hist, shrinkage=0.5)
    expected = 0.5 * (mu_pred + hist.mean()) + 0.5 * hist
    np.testing.assert_allclose(out.values, expected.values)
