"""Membership inference audit of the production model, with an overfit control.

Exit code 1 if the production model's risk is above the release gate.
"""

import json
import sys
from pathlib import Path

import yaml

from riskline import data
from riskline.privacy import GATE, audit
from riskline.registry import Registry
from riskline.schema import FEATURES, TARGET
from riskline.train import build, split

# Same features, no regularization: deep trees, single-row leaves, no early stopping.
OVERFIT = {
    "learning_rate": 0.1,
    "max_iter": 500,
    "max_leaf_nodes": 255,
    "min_samples_leaf": 1,
    "l2_regularization": 0.0,
    "early_stopping": False,
}


def main() -> int:
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    src, seed = cfg["data"], cfg["seed"]
    df = data.validate(data.load(data.fetch(src["url"], src["sha256"], src["cache"])))
    # Members: rows the trees were fit on. Non-members: the untouched test split.
    train, _, test = split(df, src["splits"], seed)
    model, card, _ = Registry(cfg["registry"]).load()

    candidates = {
        card["version"]: (model, build(cfg["model"], seed)),
        "overfit_control": (build(OVERFIT, seed).fit(train[FEATURES], train[TARGET]), build(OVERFIT, seed)),
    }
    results = {}
    for name, (fitted, recipe) in candidates.items():
        report = audit(fitted, recipe, train, test, TARGET, seed, verbose=True)
        results[name] = report.to_dict()
        s = report.strongest
        print(
            f"{name}: risk {report.risk}, strongest {s.name} AUC {s.auc:.3f}, TPR@1%FPR {s.tpr_at[0.01]:.3f}, "
            f"train/test acc {report.train_accuracy:.3f}/{report.test_accuracy:.3f}"
        )

    Path("reports").mkdir(exist_ok=True)
    Path("reports/privacy.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if results[card["version"]]["risk"] in GATE else 1


if __name__ == "__main__":
    sys.exit(main())
