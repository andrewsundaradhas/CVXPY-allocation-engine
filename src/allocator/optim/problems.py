from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .constraints import box, fully_invested, long_only, sector_caps, turnover_limit
from .objectives import mean_variance, min_variance


@dataclass
class OptimConfig:
    objective: str = "mean_variance"   # mean_variance | min_variance | max_sharpe | risk_parity
    risk_aversion: float = 5.0
    long_only: bool = True
    fully_invested: bool = True
    max_weight: float | None = 0.20
    min_weight: float = 0.0
    sector_matrix: np.ndarray | None = None
    sector_cap: float = 0.35
    w_prev: np.ndarray | None = None
    max_turnover: float | None = None


def solve_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    cfg: OptimConfig,
) -> dict:
    n = len(mu)
    w = cp.Variable(n)

    if cfg.objective == "mean_variance":
        obj = mean_variance(w, mu, Sigma, cfg.risk_aversion)
    elif cfg.objective == "min_variance":
        obj = min_variance(w, Sigma)
    else:
        raise ValueError(f"Use the dedicated solver for {cfg.objective}")

    cons: list = []
    if cfg.fully_invested:
        cons += fully_invested(w)
    if cfg.long_only:
        cons += long_only(w)
    if cfg.max_weight is not None:
        cons += box(w, cfg.min_weight, cfg.max_weight)
    if cfg.sector_matrix is not None:
        cons += sector_caps(w, cfg.sector_matrix, cfg.sector_cap)
    if cfg.w_prev is not None and cfg.max_turnover is not None:
        cons += turnover_limit(w, cfg.w_prev, cfg.max_turnover)

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"solver failed: {prob.status}")

    weights = np.asarray(w.value).flatten()
    weights = np.where(weights < 1e-6, 0.0, weights)
    if cfg.fully_invested and weights.sum() > 0:
        weights = weights / weights.sum()
    port_var = float(weights @ Sigma @ weights)
    port_ret = float(mu @ weights)
    return {
        "weights": weights,
        "expected_return": port_ret,
        "volatility": float(np.sqrt(port_var)),
        "sharpe": port_ret / np.sqrt(port_var) if port_var > 0 else 0.0,
        "status": prob.status,
    }


def max_sharpe_problem(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    n = len(mu)
    y = cp.Variable(n)
    k = cp.Variable(nonneg=True)
    excess = mu - rf
    constraints = [
        excess @ y == 1,
        cp.sum(y) == k,
        y >= 0,
        k >= 0,
    ]
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(Sigma))), constraints)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"max-Sharpe solver failed: {prob.status}")
    w_star = y.value / k.value
    return w_star


def risk_parity(Sigma: np.ndarray) -> np.ndarray:
    n = Sigma.shape[0]
    x = cp.Variable(n, nonneg=True)
    obj = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(Sigma)) - (1.0 / n) * cp.sum(cp.log(x)))
    prob = cp.Problem(obj)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"risk-parity solver failed: {prob.status}")
    return x.value / x.value.sum()
