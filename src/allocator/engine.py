from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data.features import build_features, forward_return
from .data.loader import load_prices
from .ml.blend import blend_mu
from .ml.return_model import ExpectedReturnModel
from .optim.problems import OptimConfig, solve_portfolio
from .stats.covariance import annualize_cov, ledoit_wolf_cov
from .stats.returns import log_returns


@dataclass
class AllocationRequest:
    tickers: list[str]
    lookback_years: int = 5
    objective: str = "mean_variance"
    risk_aversion: float = 5.0
    max_weight: float = 0.25
    use_ml: bool = True
    ml_shrinkage: float = 0.5


def run_allocation(req: AllocationRequest) -> dict:
    end = pd.Timestamp.today().normalize()
    start = (end - pd.DateOffset(years=req.lookback_years)).date().isoformat()
    prices = load_prices(req.tickers, start=start)
    rets = log_returns(prices)

    Sigma = annualize_cov(ledoit_wolf_cov(rets))
    hist_mu_annual = rets.mean() * 252

    if req.use_ml and len(rets) > 300:
        feats = build_features(prices, rets)
        fwd = forward_return(rets, 21)
        model = ExpectedReturnModel(horizon=21).fit(feats, fwd)
        mu_pred_monthly = model.predict_latest(feats)
        hist_mu_monthly = rets.mean() * 21
        mu_blended_monthly = blend_mu(mu_pred_monthly, hist_mu_monthly, req.ml_shrinkage)
        mu = (mu_blended_monthly * 12).reindex(req.tickers).fillna(hist_mu_annual).values
    else:
        mu = hist_mu_annual.reindex(req.tickers).values

    cfg = OptimConfig(
        objective=req.objective,
        risk_aversion=req.risk_aversion,
        max_weight=req.max_weight,
    )
    result = solve_portfolio(mu, Sigma, cfg)
    result["tickers"] = list(req.tickers)
    result["mu_used"] = mu.tolist()
    # Artifacts the dashboard needs for plotting (avoid re-downloading)
    result["returns"] = rets
    result["Sigma"] = Sigma
    result["mu"] = mu
    return result
