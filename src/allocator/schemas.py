from __future__ import annotations

from pydantic import BaseModel, Field


class AllocationIn(BaseModel):
    tickers: list[str] = Field(min_length=2, max_length=100)
    lookback_years: int = 5
    objective: str = "mean_variance"
    risk_aversion: float = 5.0
    max_weight: float = 0.25
    use_ml: bool = True
    ml_shrinkage: float = 0.5


class AllocationOut(BaseModel):
    tickers: list[str]
    weights: list[float]
    expected_return: float
    volatility: float
    sharpe: float
    status: str
