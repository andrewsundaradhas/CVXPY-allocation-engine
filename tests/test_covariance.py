from __future__ import annotations

import numpy as np
import pandas as pd

from allocator.stats.covariance import annualize_cov, ledoit_wolf_cov, sample_cov


def _toy_returns(n: int = 252, k: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    cols = [f"A{i}" for i in range(k)]
    return pd.DataFrame(rng.standard_normal((n, k)) * 0.01, index=idx, columns=cols)


def test_sample_cov_shape():
    r = _toy_returns()
    S = sample_cov(r)
    assert S.shape == (5, 5)
    np.testing.assert_allclose(S, S.T)


def test_ledoit_wolf_is_symmetric_and_psd():
    r = _toy_returns()
    Sigma = ledoit_wolf_cov(r)
    assert Sigma.shape == (5, 5)
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(Sigma)
    assert eigs.min() > 0


def test_ledoit_wolf_psd_with_few_observations():
    """Sample cov degenerate when n_obs <= n_assets, but Ledoit-Wolf must stay PSD."""
    r = _toy_returns(n=10, k=20, seed=1)
    Sigma = ledoit_wolf_cov(r)
    eigs = np.linalg.eigvalsh(Sigma)
    assert eigs.min() > 0


def test_annualize_cov_scales_by_periods():
    Sigma = np.array([[1e-4, 2e-5], [2e-5, 1e-4]])
    Sa = annualize_cov(Sigma, periods=252)
    np.testing.assert_allclose(Sa, Sigma * 252)
