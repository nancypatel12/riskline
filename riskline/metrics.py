"""Evaluation: discrimination, calibration, and cost."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(
        sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    )


def expected_cost(y, p, threshold: float, c_fn: float, c_fp: float) -> float:
    """Mean cost per applicant when flagging p >= threshold."""
    y, flag = np.asarray(y), np.asarray(p) >= threshold
    return float((c_fn * (~flag & (y == 1)) + c_fp * (flag & (y == 0))).mean())


def best_threshold(y, p, c_fn: float, c_fp: float) -> float:
    """Threshold that minimizes expected cost on the given data."""
    grid = np.linspace(0.01, 0.99, 99)
    costs = [expected_cost(y, p, t, c_fn, c_fp) for t in grid]
    return round(float(grid[int(np.argmin(costs))]), 2)


def evaluate(y, p, threshold: float, c_fn: float, c_fp: float) -> dict[str, float]:
    y, flag = np.asarray(y), np.asarray(p) >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": ece(y, p),
        "recall": float(flag[y == 1].mean()),
        "precision": float(y[flag].mean()) if flag.any() else 0.0,
        "cost": expected_cost(y, p, threshold, c_fn, c_fp),
        "cost_at_0.5": expected_cost(y, p, 0.5, c_fn, c_fp),
    }
