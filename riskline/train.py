"""Reproducible training: fetch -> validate -> split -> fit -> calibrate -> evaluate -> register."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from riskline import data
from riskline.features import FeatureBuilder
from riskline.metrics import best_threshold, evaluate
from riskline.registry import Registry
from riskline.schema import CATEGORICAL, FEATURES, TARGET


def split(df: pd.DataFrame, fractions: dict, seed: int):
    """Stratified train / calibration / test split."""
    rest, test = train_test_split(df, test_size=fractions["test"], stratify=df[TARGET], random_state=seed)
    calib_share = fractions["calib"] / (fractions["train"] + fractions["calib"])
    train, calib = train_test_split(rest, test_size=calib_share, stratify=rest[TARGET], random_state=seed)
    return train, calib, test


def build(params: dict, seed: int) -> Pipeline:
    params = {k: v for k, v in params.items() if k != "calibration"}
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            ("model", HistGradientBoostingClassifier(categorical_features=CATEGORICAL, random_state=seed, **params)),
        ]
    )


def fit(df: pd.DataFrame, cfg: dict):
    """Train, calibrate and evaluate. Returns (model, card, reference)."""
    seed, cost = cfg["seed"], cfg["cost"]
    c_fn, c_fp = cost["false_negative"], cost["false_positive"]
    train, calib, test = split(df, cfg["data"]["splits"], seed)
    Xtr, ytr = train[FEATURES], train[TARGET]
    Xca, yca = calib[FEATURES], calib[TARGET]
    Xte, yte = test[FEATURES], test[TARGET]

    gbm = build(cfg["model"], seed).fit(Xtr, ytr)
    model = CalibratedClassifierCV(FrozenEstimator(gbm), method=cfg["model"]["calibration"]).fit(Xca, yca)
    baseline = make_pipeline(FeatureBuilder(), StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)

    def score(m):
        # Thresholds are chosen on calibration data, never on test.
        t = best_threshold(yca, m.predict_proba(Xca)[:, 1], c_fn, c_fp)
        return {"threshold": t, **evaluate(yte, m.predict_proba(Xte)[:, 1], t, c_fn, c_fp)}

    metrics = score(model)
    threshold = metrics.pop("threshold")
    card = {
        "threshold": threshold,
        "cost": cost,
        "features": FEATURES,
        "rows": {"train": len(train), "calib": len(calib), "test": len(test)},
        "default_rate": float(df[TARGET].mean()),
        "metrics": metrics,
        "comparison": {
            "logistic_regression": score(baseline),
            "gbm_uncalibrated": score(gbm),
            "gbm_calibrated": {"threshold": threshold, **metrics},
        },
    }
    return model, card, Xtr


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--report", default="reports/metrics.json")
    args = parser.parse_args(argv)

    raw = Path(args.config).read_bytes()
    cfg = yaml.safe_load(raw)
    src = cfg["data"]
    df = data.validate(data.load(data.fetch(src["url"], src["sha256"], src["cache"])))

    model, card, reference = fit(df, cfg)
    card |= {
        "seed": cfg["seed"],
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "data_sha256": src["sha256"],
        "git_commit": _git_commit(),
    }
    version, promoted = Registry(cfg["registry"]).register(model, card, reference)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"version": version, **card}, indent=2) + "\n")

    print(f"{version} {'promoted to production' if promoted else 'registered (not promoted)'}")
    print(f"threshold {card['threshold']:.2f}")
    print(f"{'model':<22}{'ROC-AUC':>9}{'PR-AUC':>9}{'Brier':>8}{'ECE':>8}{'thresh':>8}{'cost':>8}")
    for name, m in card["comparison"].items():
        print(
            f"{name:<22}{m['roc_auc']:>9.4f}{m['pr_auc']:>9.4f}{m['brier']:>8.4f}"
            f"{m['ece']:>8.4f}{m['threshold']:>8.2f}{m['cost']:>8.4f}"
        )


if __name__ == "__main__":
    main()
