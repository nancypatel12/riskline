import numpy as np
import pandas as pd
import pytest
import yaml

from riskline.registry import Registry
from riskline.schema import DELAYS, MONTHS, TARGET
from riskline.train import fit


def synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Schema-valid data where delays and utilization drive default, like the real set."""
    rng = np.random.default_rng(seed)
    limit = rng.choice([20_000, 50_000, 100_000, 200_000, 500_000], n)
    df = pd.DataFrame(
        {
            "limit_bal": limit,
            "sex": rng.integers(1, 3, n),
            "education": rng.integers(0, 7, n),
            "marriage": rng.integers(0, 4, n),
            "age": rng.integers(21, 70, n),
        }
    )
    risk = rng.normal(size=n)
    for d in DELAYS:
        df[d] = np.clip(np.round(risk + rng.normal(scale=0.8, size=n)), -2, 8).astype(int)
    for m in MONTHS:
        df[f"bill_amt{m}"] = (limit * rng.uniform(0, 1, n)).round()
        df[f"pay_amt{m}"] = (df[f"bill_amt{m}"] * rng.uniform(0, 0.3, n)).round()
    logit = 1.2 * risk + 1.5 * df["bill_amt1"] / limit - 2
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture(scope="session")
def cfg():
    with open("config.yaml") as f:
        c = yaml.safe_load(f)
    c["model"]["max_iter"] = 100
    return c


@pytest.fixture(scope="session")
def df():
    return synthetic()


@pytest.fixture(scope="session")
def trained(df, cfg):
    return fit(df, cfg)


@pytest.fixture(scope="session")
def registry(trained, tmp_path_factory):
    model, card, reference = trained
    reg = Registry(tmp_path_factory.mktemp("models"))
    reg.register(model, card, reference)
    return reg
