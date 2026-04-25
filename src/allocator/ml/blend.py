from __future__ import annotations

import pandas as pd


def blend_mu(
    mu_pred: pd.Series,
    historical_mean: pd.Series,
    shrinkage: float = 0.5,
) -> pd.Series:
    """
    mu_pred:           cross-sectionally demeaned ML prediction (monthly).
    historical_mean:   per-asset historical monthly mean.
    shrinkage:         0.0 = trust the model fully, 1.0 = ignore the model.
    """
    grand_mean = historical_mean.mean()
    mu_model_abs = mu_pred + grand_mean
    prior = historical_mean
    mu_blend = (1 - shrinkage) * mu_model_abs + shrinkage * prior
    return mu_blend
