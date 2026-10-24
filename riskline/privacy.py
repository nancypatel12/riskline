"""Membership inference audit via trainleak: can an attacker tell who was in the training set?"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import roc_auc_score

from riskline.schema import FEATURES, INTEGER

GATE = ("low", "moderate")  # risk levels allowed to ship


def frame(X) -> pd.DataFrame:
    """trainleak passes numpy arrays; the pipeline needs named, typed columns."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X), columns=FEATURES).astype({c: int for c in INTEGER})


class Framed(ClassifierMixin, BaseEstimator):
    """Fits a fresh copy of `estimator` on numpy input, for LiRA reference models."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.model_ = clone(self.estimator).fit(frame(X), y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(frame(X))


def audit(model, recipe, members: pd.DataFrame, nonmembers: pd.DataFrame, target: str, seed: int, **kwargs):
    """Audit `model`, trained on `members`. `recipe` is an unfitted estimator that reproduces it."""
    from trainleak import audit as run

    return run(
        model,
        members[FEATURES],
        members[target],
        nonmembers[FEATURES],
        nonmembers[target],
        predict_proba=lambda m, X: m.predict_proba(frame(X)),
        model_factory=lambda: Framed(recipe),
        seed=seed,
        **kwargs,
    )


def sweep(params: dict, rounds, members: pd.DataFrame, nonmembers: pd.DataFrame, target: str, seed: int, **kwargs):
    """Audit one model per training length. Early stopping is off, so every model trains exactly `n` rounds."""
    from riskline.train import build

    rows = []
    for n in rounds:
        recipe = build({**params, "max_iter": n, "early_stopping": False}, seed)
        fitted = clone(recipe).fit(members[FEATURES], members[target])
        report = audit(fitted, recipe, members, nonmembers, target, seed, **kwargs)
        s = report.strongest
        scores = fitted.predict_proba(nonmembers[FEATURES])[:, 1]
        rows.append(
            {
                "rounds": n,
                "test_roc_auc": float(roc_auc_score(nonmembers[target], scores)),
                "train_accuracy": report.train_accuracy,
                "test_accuracy": report.test_accuracy,
                "strongest_attack": s.name,
                "attack_auc": s.auc,
                "tpr_at_1pct_fpr": s.tpr_at[0.01],
                "risk": report.risk,
            }
        )
    return rows
