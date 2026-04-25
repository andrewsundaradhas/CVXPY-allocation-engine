from __future__ import annotations

import numpy as np
import pandas as pd

from allocator.stats.returns import log_returns, simple_returns


def _toy_prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {"A": [100.0, 101.0, 102.01, 100.99, 102.0], "B": [50.0, 50.5, 50.0, 49.5, 50.0]},
        index=idx,
    )


def test_log_returns_drop_first_row():
    prices = _toy_prices()
    r = log_returns(prices)
    assert len(r) == len(prices) - 1
    assert not r.isna().any().any()


def test_log_returns_are_time_additive():
    prices = _toy_prices()
    r = log_returns(prices)
    summed = r.sum()
    direct = np.log(prices.iloc[-1] / prices.iloc[0])
    pd.testing.assert_series_equal(summed, direct, check_names=False)


def test_simple_returns_match_pct_change():
    prices = _toy_prices()
    r = simple_returns(prices)
    expected = prices.pct_change().dropna()
    pd.testing.assert_frame_equal(r, expected)
