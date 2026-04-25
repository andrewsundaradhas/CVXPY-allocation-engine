from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from allocator.data.loader import _cache_key, load_prices


def _fake_multi_ticker_frame(tickers: list[str], n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    cols = pd.MultiIndex.from_product([tickers, ["Close"]])
    data = 100 + rng.standard_normal((n, len(tickers))).cumsum(axis=0)
    frame = pd.DataFrame(np.repeat(data, 1, axis=1), index=idx, columns=cols)
    return frame


def test_cache_key_is_deterministic_and_order_independent(tmp_path, monkeypatch):
    monkeypatch.setattr("allocator.data.loader.CACHE_DIR", tmp_path)
    a = _cache_key(["AAPL", "MSFT"], "2024-01-01", "2024-12-31")
    b = _cache_key(["MSFT", "AAPL"], "2024-01-01", "2024-12-31")
    c = _cache_key(["AAPL", "GOOG"], "2024-01-01", "2024-12-31")
    assert a == b
    assert a != c


def test_load_prices_round_trips_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("allocator.data.loader.CACHE_DIR", tmp_path)
    tickers = ["AAPL", "MSFT"]
    fake = _fake_multi_ticker_frame(tickers)

    with patch("allocator.data.loader.yf.download", return_value=fake) as dl:
        first = load_prices(tickers, "2024-01-01", "2024-02-01", use_cache=True)
        second = load_prices(tickers, "2024-01-01", "2024-02-01", use_cache=True)

    assert dl.call_count == 1
    assert list(first.columns) == tickers
    assert first.shape[0] > 0
    pd.testing.assert_frame_equal(first, second)
