import numpy as np
import pandas as pd
import pytest

from riskline.fairness import by_group, groups


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
