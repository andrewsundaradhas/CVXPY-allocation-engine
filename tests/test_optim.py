from __future__ import annotations

import numpy as np
import pytest

from allocator.optim.problems import (
    OptimConfig,
    max_sharpe_problem,
    risk_parity,
    solve_portfolio,
)


def _toy_inputs(n: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T / n + 0.01 * np.eye(n)
    mu = np.linspace(0.05, 0.15, n)
    return mu, Sigma


def test_mean_variance_basic_invariants():
    mu, Sigma = _toy_inputs()
    out = solve_portfolio(mu, Sigma, OptimConfig(objective="mean_variance", risk_aversion=3.0))
    w = out["weights"]
    assert len(w) == 5
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-9).all()
    assert out["status"] in ("optimal", "optimal_inaccurate")
    assert out["volatility"] > 0


def test_low_risk_aversion_chases_highest_mu():
    mu, Sigma = _toy_inputs()
    out = solve_portfolio(mu, Sigma, OptimConfig(
        objective="mean_variance", risk_aversion=0.1, max_weight=1.0,
    ))
    # With near-zero risk aversion and no upper cap, the highest-mu asset should dominate.
    assert int(np.argmax(out["weights"])) == int(np.argmax(mu))


def test_max_weight_enforced():
    mu, Sigma = _toy_inputs()
    cfg = OptimConfig(objective="mean_variance", risk_aversion=0.5, max_weight=0.3)
    out = solve_portfolio(mu, Sigma, cfg)
    assert (out["weights"] <= 0.3 + 1e-6).all()


def test_min_variance_is_lower_vol_than_mean_variance():
    mu, Sigma = _toy_inputs()
    mv = solve_portfolio(mu, Sigma, OptimConfig(objective="min_variance", max_weight=1.0))
    meanv = solve_portfolio(mu, Sigma, OptimConfig(
        objective="mean_variance", risk_aversion=0.1, max_weight=1.0,
    ))
    assert mv["volatility"] <= meanv["volatility"] + 1e-9


def test_sector_cap_enforced():
    mu, Sigma = _toy_inputs(n=4)
    sector = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    cfg = OptimConfig(
        objective="mean_variance",
        risk_aversion=0.5,
        max_weight=1.0,
        sector_matrix=sector,
        sector_cap=0.55,
    )
    out = solve_portfolio(mu, Sigma, cfg)
    exposures = sector @ out["weights"]
    assert (exposures <= 0.55 + 1e-6).all()


def test_turnover_limit_keeps_solution_near_prev():
    mu, Sigma = _toy_inputs()
    w_prev = np.full(5, 0.2)
    cfg = OptimConfig(
        objective="mean_variance",
        risk_aversion=0.5,
        max_weight=1.0,
        w_prev=w_prev,
        max_turnover=0.1,
    )
    out = solve_portfolio(mu, Sigma, cfg)
    assert np.linalg.norm(out["weights"] - w_prev, ord=1) <= 0.1 + 1e-6


def test_unsupported_objective_routes_to_dedicated_solver():
    mu, Sigma = _toy_inputs()
    with pytest.raises(ValueError, match="dedicated solver"):
        solve_portfolio(mu, Sigma, OptimConfig(objective="max_sharpe"))


def test_max_sharpe_basic_invariants():
    mu, Sigma = _toy_inputs()
    w = max_sharpe_problem(mu, Sigma, rf=0.0)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-4)
    assert (w >= -1e-5).all()


def test_risk_parity_equal_contributions():
    _, Sigma = _toy_inputs()
    w = risk_parity(Sigma)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
    assert (w > 0).all()
    contribs = w * (Sigma @ w)
    contribs = contribs / contribs.sum()
    np.testing.assert_allclose(contribs, np.full_like(contribs, 1.0 / len(contribs)), atol=0.02)
