from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(".cache/prices")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(tickers: list[str], start: str, end: str) -> Path:
    raw = ",".join(sorted(tickers)) + f"|{start}|{end}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return CACHE_DIR / f"prices_{h}.parquet"


def load_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Adjusted close prices, one column per ticker, NaN-dropped."""
    if end is None:
        end = date.today().isoformat()
    if start is None:
        start = (date.today() - timedelta(days=365 * 5)).isoformat()

    cache_path = _cache_key(tickers, start, end)
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if isinstance(raw.columns, pd.MultiIndex):
        available = raw.columns.get_level_values(0).unique()
        prices = pd.concat(
            {t: raw[t]["Close"] for t in tickers if t in available},
            axis=1,
        )
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill().dropna()
    prices.to_parquet(cache_path)
    return prices
