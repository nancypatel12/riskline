import numpy as np
import pandas as pd

from riskline.drift import psi, report


def test_psi_zero_for_same_distribution():
    x = np.random.default_rng(0).normal(size=20_000)
    assert psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref, cur = rng.normal(size=20_000), rng.normal(0.5, size=20_000)
    assert psi(ref, cur) > 0.2


def test_psi_handles_discrete_features():
    rng = np.random.default_rng(0)
    ref = rng.integers(-2, 3, 10_000)
    assert psi(ref, rng.integers(-2, 3, 10_000)) < 0.01
    assert psi(ref, np.clip(ref + 1, -2, 8)) > 0.2


def test_report_flags_only_shifted_feature():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.integers(0, 4, 5000)})
    cur = pd.DataFrame({"a": rng.normal(1.0, size=5000), "b": rng.integers(0, 4, 5000)})
    r = report(ref, cur).set_index("feature")
    assert r.loc["a", "status"] == "alert"
    assert r.loc["b", "status"] == "ok"
