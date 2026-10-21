"""Per-feature drift: Population Stability Index and Kolmogorov-Smirnov."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-6
DISCRETE = 20  # features with at most this many distinct values are binned by value


def psi(reference, current, bins: int = 10) -> float:
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    values = np.unique(ref)
    if len(values) <= DISCRETE:
        edges = np.concatenate([[-np.inf], (values[:-1] + values[1:]) / 2, [np.inf]])
    else:
        inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
        edges = np.concatenate([[-np.inf], inner, [np.inf]])
    r = np.histogram(ref, edges)[0] / len(ref) + EPS
    c = np.histogram(cur, edges)[0] / len(cur) + EPS
    return float(np.sum((c - r) * np.log(c / r)))


def report(reference: pd.DataFrame, current: pd.DataFrame, psi_warn=0.1, psi_alert=0.2, ks_alert=0.1) -> pd.DataFrame:
    rows = []
    for col in reference.columns:
        p = psi(reference[col], current[col])
        ks = float(ks_2samp(reference[col], current[col]).statistic)
        status = "alert" if p >= psi_alert or ks >= ks_alert else "warn" if p >= psi_warn else "ok"
        rows.append({"feature": col, "psi": round(p, 4), "ks": round(ks, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
