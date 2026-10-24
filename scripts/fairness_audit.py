"""Outcomes by sex, age and marital status for the production model, on the test split."""

import json
from pathlib import Path

import yaml

from riskline import data
from riskline.fairness import by_group, groups
from riskline.registry import Registry
from riskline.schema import FEATURES, TARGET
from riskline.train import split


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
        results["attributes"][attr] = report
        print(attr)
        for g, r in report["groups"].items():
            print(
                f"  {g:<12}{r['n']:>6}  default {r['default_rate']:.3f}  score {r['mean_score']:.3f}  "
                f"flagged {r['flag_rate']:.3f}  good wrongly flagged {r['wrongly_flagged']:.3f}  "
                f"defaults caught {r['defaults_caught']:.3f}"
            )
        print("  gaps " + "  ".join(f"{k} {v:.3f}" for k, v in report["gaps"].items()))

    Path("reports").mkdir(exist_ok=True)
    Path("reports/fairness.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
