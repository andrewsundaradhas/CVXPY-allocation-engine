from __future__ import annotations

from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from allocator.api import app

client = TestClient(app)


def _stub_result(tickers: list[str]) -> dict:
    n = len(tickers)
    return {
        "tickers": list(tickers),
        "weights": np.full(n, 1.0 / n),
        "expected_return": 0.08,
        "volatility": 0.15,
        "sharpe": 0.53,
        "status": "optimal",
        "mu_used": [0.08] * n,
    }


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_allocate_happy_path():
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "PG"]
    with patch("allocator.api.run_allocation", return_value=_stub_result(tickers)):
        r = client.post(
            "/allocate",
            json={
                "tickers": tickers,
                "objective": "mean_variance",
                "risk_aversion": 4.0,
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["tickers"] == tickers
    assert len(body["weights"]) == len(tickers)
    assert abs(sum(body["weights"]) - 1.0) < 1e-6
    assert body["status"] == "optimal"


def test_allocate_validates_min_tickers():
    r = client.post("/allocate", json={"tickers": ["AAPL"]})
    assert r.status_code == 422


def test_allocate_returns_400_on_engine_error():
    with patch("allocator.api.run_allocation", side_effect=RuntimeError("solver failed")):
        r = client.post("/allocate", json={"tickers": ["A", "B"]})
    assert r.status_code == 400
    assert "solver failed" in r.json()["detail"]
