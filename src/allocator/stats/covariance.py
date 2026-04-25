from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().values


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Shrunk covariance estimator. Returns numpy array (n, n)."""
    lw = LedoitWolf().fit(returns.values)
    Sigma = lw.covariance_
    Sigma = (Sigma + Sigma.T) / 2.0
    Sigma += 1e-10 * np.eye(Sigma.shape[0])
    return Sigma


def annualize_cov(Sigma_daily: np.ndarray, periods: int = 252) -> np.ndarray:
    return Sigma_daily * periods
