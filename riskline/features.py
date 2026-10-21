"""Feature engineering as a stateless, picklable sklearn transformer."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from riskline.schema import DELAYS, FEATURES, MONTHS


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Cleans undocumented category codes and adds credit-behavior ratios."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[FEATURES].copy()
        # Codes 0, 5, 6 are undocumented in the source; fold them into "other".
        X["education"] = X["education"].where(X["education"].between(1, 4), 4)
        X["marriage"] = X["marriage"].where(X["marriage"].between(1, 3), 3)

        bills = X[[f"bill_amt{m}" for m in MONTHS]].to_numpy(float)
        paid = X[[f"pay_amt{m}" for m in MONTHS]].to_numpy(float)
        limit = X["limit_bal"].to_numpy(float)
        delays = X[DELAYS].to_numpy()

        X["utilization"] = bills[:, 0] / limit
        X["avg_utilization"] = bills.mean(axis=1) / limit
        X["repay_ratio"] = np.clip(paid[:, 0] / np.maximum(bills[:, 0], 1.0), 0, 10)
        X["max_delay"] = delays.max(axis=1)
        X["months_late"] = (delays > 0).sum(axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        extra = ["utilization", "avg_utilization", "repay_ratio", "max_delay", "months_late"]
        return np.array(FEATURES + extra)
