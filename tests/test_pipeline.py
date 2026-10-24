import numpy as np
import pytest

from riskline import data
from riskline.features import FeatureBuilder
from riskline.metrics import best_threshold, ece, expected_cost
from riskline.schema import FEATURES, TARGET
from riskline.train import fit, split


def test_validate_accepts_clean_data(df):
    assert data.validate(df) is df


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.drop(columns="age"), "missing columns"),
        (lambda d: d.assign(age=d.age.astype(float).mask(d.index < 5)), "nulls"),
        (lambda d: d.assign(age=200), "outside"),
        (lambda d: d.assign(**{TARGET: 2}), "labels"),
        (lambda d: d.head(10), "rows"),
    ],
)
def test_validate_rejects_bad_data(df, mutate, message):
    with pytest.raises(data.DataValidationError, match=message):
        data.validate(mutate(df.copy()))


def test_split_is_stratified_and_disjoint(df, cfg):
    parts = split(df, cfg["data"]["splits"], seed=1)
    assert sum(map(len, parts)) == len(df)
    assert not set.intersection(*(set(p.index) for p in parts))
    rates = [p[TARGET].mean() for p in parts]
    assert max(rates) - min(rates) < 0.02


def test_features_fold_unknown_codes(df):
    X = FeatureBuilder().fit_transform(df[FEATURES])
    assert set(X.education) <= {1, 2, 3, 4}
    assert not set(X.columns) & {"sex", "age", "marriage"}
    assert np.isfinite(X.select_dtypes("number").to_numpy()).all()


def test_training_learns_and_is_deterministic(trained, df, cfg):
    _, card, _ = trained
    m = card["metrics"]
    assert m["roc_auc"] > 0.7
    assert m["ece"] < 0.05
    assert m["cost"] <= m["cost_at_0.5"]
    _, again, _ = fit(df, cfg)
    assert again["metrics"] == m


def test_threshold_matches_bayes_rule_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    # Optimal flag threshold is c_fp / (c_fp + c_fn).
    assert best_threshold(y, p, c_fn=5, c_fp=1) == pytest.approx(1 / 6, abs=0.02)
    assert ece(y, p) < 0.01
    assert expected_cost(y, p, 1.01, c_fn=5, c_fp=1) == pytest.approx(5 * y.mean())


def test_registry_roundtrip_and_gate(registry, trained):
    model, card, reference = trained
    loaded, saved, _ref = registry.load()
    assert saved["version"] == "v1"
    X = reference.head(50)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))

    worse = {**card, "metrics": {**card["metrics"], "pr_auc": 0.0}}
    version, promoted = registry.register(model, worse, reference)
    assert (version, promoted) == ("v2", False)
    assert registry.load()[1]["version"] == "v1"
