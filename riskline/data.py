"""Fetch, verify, load and validate the raw dataset."""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from riskline.schema import COLUMNS, RANGES, TARGET


class DataValidationError(ValueError):
    pass


def fetch(url: str, sha256: str, cache: str | Path) -> Path:
    """Download once, then verify the checksum every time."""
    path = Path(cache)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise DataValidationError(f"checksum mismatch for {path}: {digest}")
    return path


def load(path: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        df = pd.read_excel(f, header=1)
    df = df.rename(columns=str.lower).rename(columns={"pay_0": "pay_1", "default payment next month": TARGET})
    return df.drop(columns="id")[[*COLUMNS, TARGET]]


def validate(df: pd.DataFrame, min_rows: int = 100) -> pd.DataFrame:
    """Fail loudly, listing every violation at once."""
    errors = []
    missing = set([*COLUMNS, TARGET]) - set(df.columns)
    if missing:
        raise DataValidationError(f"missing columns: {sorted(missing)}")
    if len(df) < min_rows:
        errors.append(f"only {len(df)} rows (< {min_rows})")
    for col in [*COLUMNS, TARGET]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{col}: non-numeric dtype {df[col].dtype}")
        elif df[col].isna().any():
            errors.append(f"{col}: {int(df[col].isna().sum())} nulls")
    for col, (lo, hi) in RANGES.items():
        if pd.api.types.is_numeric_dtype(df[col]):
            bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if bad:
                errors.append(f"{col}: {bad} values outside [{lo}, {hi}]")
    if not set(df[TARGET].unique()) <= {0, 1}:
        errors.append(f"{TARGET}: labels must be 0/1")
    elif df[TARGET].nunique() < 2:
        errors.append(f"{TARGET}: only one class present")
    if errors:
        raise DataValidationError("; ".join(errors))
    return df
