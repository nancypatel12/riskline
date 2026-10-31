"""Sigmoid vs. isotonic calibration on the same splits: why the config uses sigmoid."""

import copy
import json
from pathlib import Path

import yaml

from riskline import data
from riskline.train import fit

SEEDS = (42, 43, 44)
METRICS = ("roc_auc", "pr_auc", "brier", "ece")


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    src = cfg["data"]
    df = data.validate(data.load(data.fetch(src["url"], src["sha256"], src["cache"])))

    results = {"seeds": {}}
    for seed in SEEDS:
        row = {}
        for method in ("sigmoid", "isotonic"):
            run = copy.deepcopy(cfg)
            run["seed"], run["model"]["calibration"] = seed, method
            metrics = fit(df, run)[1]["metrics"]
            row[method] = {k: metrics[k] for k in METRICS}
        # Positive means isotonic loses PR-AUC against sigmoid.
        row["pr_auc_loss_points"] = 100 * (row["sigmoid"]["pr_auc"] - row["isotonic"]["pr_auc"])
        results["seeds"][str(seed)] = row
        print(f"seed {seed}: isotonic loses {row['pr_auc_loss_points']:.2f} points of PR-AUC")

    Path("reports/calibration.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
