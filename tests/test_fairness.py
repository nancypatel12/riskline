import numpy as np
import pandas as pd
import pytest

from riskline.fairness import by_group, gap_interval, groups


def test_groups_label_every_row():
    df = pd.DataFrame({"sex": [1, 2], "age": [25, 61], "marriage": [0, 2]})
    g = groups(df)
    assert list(g["sex"]) == ["male", "female"]
    assert list(g["age"]) == ["18-29", "50+"]
    assert list(g["marriage"]) == ["other", "single"]


def test_by_group_rates_and_gaps():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    p = np.array([0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.1])
    labels = pd.Series(["a"] * 4 + ["b"] * 4)
    report = by_group(y, p, 0.5, labels)
    a, b = report["groups"]["a"], report["groups"]["b"]
    assert a["wrongly_flagged"] == 0.5 and b["wrongly_flagged"] == 0.0
    assert a["defaults_caught"] == 1.0 and b["defaults_caught"] == 0.5
    assert report["gaps"]["wrongly_flagged"] == pytest.approx(0.5)


def test_gap_interval_brackets_a_clear_gap_and_is_reproducible():
    # Group a wrongly flags every good customer, group b none: the gap is 1 however rows are resampled.
    y = np.zeros(200)
    p = np.array([0.9] * 100 + [0.1] * 100)
    labels = np.array(["a"] * 100 + ["b"] * 100)
    low, high = gap_interval(y, p, 0.5, labels, "a", "b", n=200, seed=1)
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)
    first = gap_interval(y, p, 0.5, labels, "a", "b", n=50, seed=3)
    assert first == gap_interval(y, p, 0.5, labels, "a", "b", n=50, seed=3)
