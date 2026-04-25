from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


class ExpectedReturnModel:
    """Predict next-period return from features. Cross-sectionally demeaned target."""

    def __init__(self, horizon: int = 21):
        self.horizon = horizon
        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            random_state=42,
        )

    def _xy(self, features: pd.DataFrame, fwd: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        target = fwd.stack().rename("y")
        target.index.names = features.index.names
        df = features.join(target, how="inner").dropna()
        df["y"] = df.groupby(level="date")["y"].transform(lambda s: s - s.mean())
        X = df.drop(columns=["y"]).values
        y = df["y"].values
        return X, y

    def fit(self, features: pd.DataFrame, fwd: pd.DataFrame) -> "ExpectedReturnModel":
        X, y = self._xy(features, fwd)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict_latest(self, features: pd.DataFrame) -> pd.Series:
        latest_date = features.index.get_level_values("date").max()
        latest = features.xs(latest_date, level="date")
        Xs = self.scaler.transform(latest.values)
        pred = self.model.predict(Xs)
        return pd.Series(pred, index=latest.index, name="mu_pred")
