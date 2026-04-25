from __future__ import annotations

import numpy as np
import pandas as pd

from allocator.data.features import build_features, forward_return
from allocator.ml.return_model import ExpectedReturnModel


def _toy_prices(n: int = 500, k: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    cols = [f"A{i}" for i in range(k)]
    daily = rng.standard_normal((n, k)) * 0.01
    return pd.DataFrame(100 * np.exp(np.cumsum(daily, axis=0)), index=idx, columns=cols)


def test_fit_predict_latest_returns_one_per_ticker():
    prices = _toy_prices()
    rets = np.log(prices / prices.shift(1)).dropna()
    feats = build_features(prices, rets)
    fwd = forward_return(rets, 21)

    model = ExpectedReturnModel(horizon=21).fit(feats, fwd)
    pred = model.predict_latest(feats)

    assert isinstance(pred, pd.Series)
    assert pred.name == "mu_pred"
    assert len(pred) == prices.shape[1]
    assert set(pred.index) == set(prices.columns)
    assert np.isfinite(pred.values).all()
