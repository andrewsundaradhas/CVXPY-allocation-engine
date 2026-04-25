from __future__ import annotations

import cvxpy as cp
import numpy as np


def fully_invested(w: cp.Variable) -> list:
    return [cp.sum(w) == 1]


def long_only(w: cp.Variable) -> list:
    return [w >= 0]


def box(w: cp.Variable, lower: float = 0.0, upper: float = 0.2) -> list:
    return [w >= lower, w <= upper]


def sector_caps(
    w: cp.Variable,
    sector_matrix: np.ndarray,
    cap: float = 0.35,
) -> list:
    return [sector_matrix @ w <= cap]


def turnover_limit(
    w: cp.Variable,
    w_prev: np.ndarray,
    max_turnover: float = 0.2,
) -> list:
    return [cp.norm(w - w_prev, 1) <= max_turnover]
