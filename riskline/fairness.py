"""Outcomes by group: does the model treat sex, age and marital status groups alike?"""

import numpy as np
import pandas as pd


def groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Readable group labels for each protected attribute."""
    return {
        "sex": df["sex"].map({1: "male", 2: "female"}),
        "age": pd.cut(df["age"], [0, 29, 39, 49, 200], labels=["18-29", "30-39", "40-49", "50+"]).astype(str),
        # Codes 0 and 3 are undocumented or "other" in the source.
        "marriage": df["marriage"].map({1: "married", 2: "single"}).fillna("other"),
    }


def by_group(y, p, threshold: float, labels: pd.Series) -> dict:
    """Per-group rates, plus the largest gap between groups for each rate."""
    y, p, labels = np.asarray(y), np.asarray(p), np.asarray(labels)
    flag = p >= threshold
    rows = {}
    for g in sorted(set(labels)):
        m = labels == g
        good, bad = m & (y == 0), m & (y == 1)
        rows[g] = {
            "n": int(m.sum()),
            "default_rate": float(y[m].mean()),
            "mean_score": float(p[m].mean()),
            "flag_rate": float(flag[m].mean()),
            # Good customers the model would turn away.
            "wrongly_flagged": float(flag[good].mean()),
            # Defaults the model catches.
            "defaults_caught": float(flag[bad].mean()),
        }
    keys = ("flag_rate", "wrongly_flagged", "defaults_caught")
    gaps = {k: max(r[k] for r in rows.values()) - min(r[k] for r in rows.values()) for k in keys}
    return {"groups": rows, "gaps": gaps}


def gap_interval(y, p, threshold: float, labels, high: str, low: str, n: int = 1000, seed: int = 0) -> list[float]:
    """95% bootstrap interval for how much more often good customers in `high` are wrongly flagged than in `low`.

    Resamples test rows with replacement, so a gap that comes from a few customers shows up as a wide interval.
    """
    y, p, labels = np.asarray(y), np.asarray(p), np.asarray(labels)
    flag = p >= threshold
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        ys, fs, ls = y[i], flag[i], labels[i]
        rate = {g: fs[(ls == g) & (ys == 0)].mean() for g in (high, low)}
        gaps.append(rate[high] - rate[low])
    return [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
