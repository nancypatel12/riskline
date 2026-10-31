"""Feature engineering as a stateless, picklable sklearn transformer."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from riskline.schema import DELAYS, FEATURES, MONTHS


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Cleans undocumented category codes and adds credit-behavior ratios."""

    # Models saved before this parameter existed load without it and use FEATURES.
    columns = None

    def __init__(self, columns=None):
        # Only the fairness audit passes other columns, to measure what leaving sex, age and marital status out costs.
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[list(self.columns or FEATURES)].copy()
        # Codes 0, 5, 6 are undocumented in the source; fold them into "other".
        X["education"] = X["education"].where(X["education"].between(1, 4), 4)

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
        return np.array(list(self.columns or FEATURES) + extra)
