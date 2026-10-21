"""Score held-out data against the training reference, then a simulated downturn.

Exit code 1 if the shifted data raises no alert, or the clean data raises one.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from riskline import data
from riskline.drift import report
from riskline.registry import Registry
from riskline.schema import DELAYS, FEATURES, MONTHS
from riskline.train import split


def downturn(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Banks cut limits, balances grow, a third of customers slip one more month."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["limit_bal"] = (out["limit_bal"] * 0.7).round()
    for m in MONTHS:
        out[f"bill_amt{m}"] = (out[f"bill_amt{m}"] * 1.3).round()
    slip = rng.random(len(out)) < 0.35
    out.loc[slip, DELAYS] = (out.loc[slip, DELAYS] + 1).clip(upper=9)
    return out


def main() -> int:
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    src, thresholds = cfg["data"], cfg["drift"]
    df = data.validate(data.load(data.fetch(src["url"], src["sha256"], src["cache"])))
    _, _, test = split(df, src["splits"], cfg["seed"])
    model, card, reference = Registry(cfg["registry"]).load()

    ref = reference.assign(score=model.predict_proba(reference[FEATURES])[:, 1])
    results = {}
    for name, cur in {"holdout": test[FEATURES], "downturn": downturn(test[FEATURES], cfg["seed"])}.items():
        cur = cur.assign(score=model.predict_proba(cur)[:, 1])
        r = report(ref, cur, **{k: thresholds[k] for k in ("psi_warn", "psi_alert", "ks_alert")})
        alerts = r[r.status == "alert"]
        results[name] = {
            "alerts": alerts.feature.tolist(),
            "mean_score": round(float(cur.score.mean()), 4),
            "flag_rate": round(float((cur.score >= card["threshold"]).mean()), 4),
            "features": r.to_dict("records"),
        }
        print(
            f"\n{name}: {len(alerts)} alerts, mean score {cur.score.mean():.3f}, "
            f"flag rate {(cur.score >= card['threshold']).mean():.1%}"
        )
        print(r.head(8).to_string(index=False))

    Path("reports").mkdir(exist_ok=True)
    Path("reports/drift.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if results["downturn"]["alerts"] and not results["holdout"]["alerts"] else 1


if __name__ == "__main__":
    sys.exit(main())
