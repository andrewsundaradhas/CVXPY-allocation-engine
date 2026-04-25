from __future__ import annotations

import cvxpy as cp
import numpy as np


def min_variance(w: cp.Variable, Sigma: np.ndarray) -> cp.Minimize:
    return cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))


def mean_variance(
    w: cp.Variable,
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_aversion: float = 5.0,
) -> cp.Minimize:
    return cp.Minimize(-mu @ w + (risk_aversion / 2.0) * cp.quad_form(w, cp.psd_wrap(Sigma)))
