from __future__ import annotations

import pandas as pd


def momentum_12_1(returns: pd.DataFrame) -> pd.DataFrame:
    cum = (1 + returns).rolling(252).apply(lambda x: x.prod(), raw=True) - 1
    skip_1m = cum.shift(21)
    return skip_1m


def short_reversal(returns: pd.DataFrame) -> pd.DataFrame:
    return -returns.rolling(21).sum()


def rolling_vol(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    return returns.rolling(window).std()


def max_drawdown(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    roll_max = prices.rolling(window, min_periods=1).max()
    dd = prices / roll_max - 1
    return dd


def build_features(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Stacked long-format feature panel: index=(date, ticker), columns=features."""
    feats = {
        "mom_12_1": momentum_12_1(returns),
        "rev_1m":   short_reversal(returns),
        "vol_60":   rolling_vol(returns, 60),
        "mdd_252":  max_drawdown(prices, 252),
    }
    panel = pd.concat(
        {name: df.stack() for name, df in feats.items()},
        axis=1,
    ).dropna()
    panel.index.names = ["date", "ticker"]
    return panel


def forward_return(returns: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    fwd = (1 + returns).rolling(horizon).apply(lambda x: x.prod(), raw=True) - 1
    return fwd.shift(-horizon)
