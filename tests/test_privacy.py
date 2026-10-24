import numpy as np
import pytest

from riskline.privacy import Framed, audit, frame, sweep
from riskline.schema import FEATURES, INTEGER, TARGET
from riskline.train import build
from tests.conftest import synthetic

pytest.importorskip("trainleak")

FAST = {"attacks": ("loss", "lira"), "n_reference": 4, "max_samples": 300}
OVERFIT = {"max_iter": 200, "max_leaf_nodes": 255, "min_samples_leaf": 1, "early_stopping": False}


def test_frame_restores_names_and_types(df):
    X = frame(df[FEATURES].to_numpy())
    assert list(X.columns) == FEATURES
    assert all(X[c].dtype.kind == "i" for c in INTEGER)


def test_framed_fits_on_numpy(df, cfg):
    model = Framed(build(cfg["model"], 0)).fit(df[FEATURES].to_numpy(), df[TARGET])
    p = model.predict_proba(df[FEATURES].to_numpy()[:10])
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_audit_separates_overfit_from_regularized(cfg):
    members, nonmembers = synthetic(600, seed=1), synthetic(600, seed=2)

    def strongest_auc(params):
        recipe = build(params, 0)
        fitted = recipe.fit(members[FEATURES], members[TARGET])
        return audit(fitted, build(params, 0), members, nonmembers, TARGET, seed=0, **FAST).strongest.auc

    assert strongest_auc(OVERFIT) > strongest_auc(cfg["model"]) + 0.1


def test_sweep_leaks_more_with_more_rounds():
    members, nonmembers = synthetic(600, seed=1), synthetic(600, seed=2)
    rows = sweep(OVERFIT, (10, 200), members, nonmembers, TARGET, seed=0, **FAST)
    assert [r["rounds"] for r in rows] == [10, 200]
    assert rows[1]["attack_auc"] > rows[0]["attack_auc"]
    assert all(0 <= r["test_roc_auc"] <= 1 for r in rows)
