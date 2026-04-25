from __future__ import annotations

import numpy as np
import pandas as pd

from allocator.data.features import (
    build_features,
    forward_return,
    max_drawdown,
    momentum_12_1,
    rolling_vol,
    short_reversal,
)


def _toy_prices(n: int = 400, k: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    cols = [f"A{i}" for i in range(k)]
    daily = rng.standard_normal((n, k)) * 0.01
    return pd.DataFrame(100 * np.exp(np.cumsum(daily, axis=0)), index=idx, columns=cols)


def _toy_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def test_momentum_skips_recent_month():
    prices = _toy_prices()
    rets = _toy_returns(prices)
    mom = momentum_12_1(rets)
    assert mom.iloc[: 252 + 21 - 1].isna().all().all()
    assert mom.iloc[252 + 21:].notna().all().all()


def test_short_reversal_signs_flip_sum():
    prices = _toy_prices()
    rets = _toy_returns(prices)
    rev = short_reversal(rets)
    direct = -rets.rolling(21).sum()
    pd.testing.assert_frame_equal(rev, direct)


def test_rolling_vol_window_size():
    prices = _toy_prices()
    rets = _toy_returns(prices)
    v = rolling_vol(rets, 60)
    assert v.iloc[:59].isna().all().all()
    assert v.iloc[59:].notna().all().all()


def test_max_drawdown_is_non_positive():
    prices = _toy_prices()
    dd = max_drawdown(prices, 252).dropna()
    assert (dd <= 1e-12).all().all()


def test_build_features_long_format():
    prices = _toy_prices()
    rets = _toy_returns(prices)
    feats = build_features(prices, rets)
    assert feats.index.names == ["date", "ticker"]
    assert set(feats.columns) == {"mom_12_1", "rev_1m", "vol_60", "mdd_252"}
    assert feats.shape[0] > 0
    assert not feats.isna().any().any()


def test_forward_return_is_shifted_backwards():
    prices = _toy_prices(n=100, k=2)
    rets = _toy_returns(prices)
    fwd = forward_return(rets, horizon=5)
    # Last `horizon` rows should be NaN because there is no future data.
    assert fwd.iloc[-5:].isna().all().all()
