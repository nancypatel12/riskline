"""Does training longer make the model leak more? Audit one model per training length, for two tree sizes.

Writes reports/leakage.json. Pass --plot to redraw reports/leakage.png from it without retraining.
"""

import argparse
import json
from pathlib import Path

import yaml

from riskline import data
from riskline.privacy import sweep
from riskline.schema import FEATURES, TARGET
from riskline.train import build, split

ROUNDS = (25, 50, 100, 200, 400, 800)

# Same learning rate and features as production; only the trees change.
TREES = {
    "production": {},
    "deep": {"max_leaf_nodes": 255, "min_samples_leaf": 1, "l2_regularization": 0.0},
}

REPORT = Path("reports/leakage.json")
CHART = Path("reports/leakage.png")


def run() -> dict:
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    src, seed = cfg["data"], cfg["seed"]
    df = data.validate(data.load(data.fetch(src["url"], src["sha256"], src["cache"])))
    # Members: rows the trees were fit on. Non-members: the untouched test split.
    train, _, test = split(df, src["splits"], seed)
    params = {k: v for k, v in cfg["model"].items() if k != "calibration"}

    # Where early stopping halts the production model, for reference on the chart.
    stopped = build(params, seed).fit(train[FEATURES], train[TARGET])[-1].n_iter_
    results = {"rounds": list(ROUNDS), "early_stopping_rounds": int(stopped), "trees": {}}
    for name, change in TREES.items():
        rows = sweep({**params, **change}, ROUNDS, train, test, TARGET, seed)
        results["trees"][name] = rows
        for r in rows:
            print(
                f"{name:<11}{r['rounds']:>5} rounds  test ROC-AUC {r['test_roc_auc']:.3f}  "
                f"attack AUC {r['attack_auc']:.3f} ({r['strongest_attack']})  TPR@1%FPR {r['tpr_at_1pct_fpr']:.3f}"
            )
    return results


def plot(results: dict) -> None:
    """Two panels on one scale: test ROC-AUC stays flat while the attack's AUC climbs."""
    # matplotlib is optional, so the sweep's numbers never depend on it.
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        print("skipped the chart: pip install -e '.[plots]', then rerun with --plot")
        return

    mpl.rcParams.update(
        {
            "font.family": ["Inter", "Helvetica Neue", "Arial"],
            "font.size": 11,
            "axes.edgecolor": "#e6e6e6",
            "axes.labelcolor": "#888888",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "xtick.color": "#888888",
            "ytick.color": "#888888",
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
    colors = {"production": "#3a4fbf", "deep": "#9a9aa3"}
    labels = {"production": "Production trees", "deep": "Deep trees"}
    # Test ROC-AUC lines end close together, so their labels move apart.
    nudge = {"production": 6, "deep": -6}
    panels = [("test_roc_auc", "Test ROC-AUC (new customers)"), ("attack_auc", "Attack AUC (who was in training)")]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (key, title) in zip(axes, panels, strict=True):
        for name, rows in results["trees"].items():
            x, y = [r["rounds"] for r in rows], [r[key] for r in rows]
            ax.plot(x, y, color=colors[name], linewidth=2, marker="o", markersize=4)
            ax.annotate(
                f"{labels[name]} {y[-1]:.2f}",
                (x[-1], y[-1]),
                xytext=(6, nudge[name]),
                textcoords="offset points",
                va="center",
                color=colors[name],
                fontsize=9,
            )
        ax.axvline(results["early_stopping_rounds"], color="#cfcfd6", linestyle=":", linewidth=1.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(results["rounds"], [str(r) for r in results["rounds"]])
        ax.minorticks_off()
        ax.set_xlabel("Boosting rounds")
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].set_ylim(0.5, 0.9)
    axes[0].annotate(
        f"Early stopping: {results['early_stopping_rounds']}",
        (results["early_stopping_rounds"], 0.52),
        xytext=(4, 0),
        textcoords="offset points",
        color="#888888",
        fontsize=9,
    )
    fig.suptitle("Training longer helps the attacker, not the model", x=0.06, ha="left", fontsize=12, fontweight=600)
    fig.tight_layout()
    fig.savefig(CHART)
    print(f"wrote {CHART}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="redraw the chart from the saved report")
    args = parser.parse_args()

    if args.plot:
        results = json.loads(REPORT.read_text())
    else:
        results = run()
        REPORT.parent.mkdir(exist_ok=True)
        REPORT.write_text(json.dumps(results, indent=2) + "\n")
    plot(results)


if __name__ == "__main__":
    main()
