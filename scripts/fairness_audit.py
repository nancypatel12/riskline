"""Outcomes by sex, age and marital status on the test split, and what leaving them out of the model costs."""

import json
from pathlib import Path

import yaml

from riskline import data
from riskline.fairness import by_group, gap_interval, groups
from riskline.registry import Registry
from riskline.schema import COLUMNS, FEATURES, TARGET
from riskline.train import fit, split


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    src, seed = cfg["data"], cfg["seed"]
    df = data.validate(data.load(data.fetch(src["url"], src["sha256"], src["cache"])))
    _, _, test = split(df, src["splits"], seed)
    model, card, _ = Registry(cfg["registry"]).load()
    p = model.predict_proba(test[FEATURES])[:, 1]

    results = {"version": card["version"], "threshold": card["threshold"], "n": len(test), "attributes": {}}
    for attr, labels in groups(test).items():
        report = by_group(test[TARGET], p, card["threshold"], labels)
        # Is the largest gap in good customers wrongly flagged bigger than chance?
        rates = {g: r["wrongly_flagged"] for g, r in report["groups"].items()}
        high, low = max(rates, key=rates.get), min(rates, key=rates.get)
        report["wrongly_flagged_gap"] = {
            "between": [high, low],
            "gap": rates[high] - rates[low],
            "interval_95": gap_interval(test[TARGET], p, card["threshold"], labels, high, low, seed=seed),
        }
        results["attributes"][attr] = report
        print(attr)
        for g, r in report["groups"].items():
            print(
                f"  {g:<12}{r['n']:>6}  default {r['default_rate']:.3f}  score {r['mean_score']:.3f}  "
                f"flagged {r['flag_rate']:.3f}  good wrongly flagged {r['wrongly_flagged']:.3f}  "
                f"defaults caught {r['defaults_caught']:.3f}"
            )
        print("  gaps " + "  ".join(f"{k} {v:.3f}" for k, v in report["gaps"].items()))
        w = report["wrongly_flagged_gap"]
        low, high = w["interval_95"]
        print(f"  wrongly flagged, {w['between'][0]} vs {w['between'][1]}: {w['gap']:.3f} [{low:.3f}, {high:.3f}]")

    # The same recipe, trained with sex, age and marital status as inputs.
    _, with_protected, _ = fit(df, cfg, columns=COLUMNS)
    results["test_roc_auc"] = {
        "without_protected": card["metrics"]["roc_auc"],
        "with_protected": with_protected["metrics"]["roc_auc"],
    }
    print("test ROC-AUC  " + "  ".join(f"{k} {v:.4f}" for k, v in results["test_roc_auc"].items()))

    Path("reports").mkdir(exist_ok=True)
    Path("reports/fairness.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
