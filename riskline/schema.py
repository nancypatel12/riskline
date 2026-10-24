"""Single source of truth for the input contract."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, create_model

MONTHS = range(1, 7)
TARGET = "default"
CATEGORICAL = ["education"]
DELAYS = [f"pay_{m}" for m in MONTHS]
INTEGER = {*CATEGORICAL, *DELAYS}

# Kept in the data to audit outcomes by group. Never given to the model or accepted by the API.
PROTECTED = ["sex", "age", "marriage"]

# name -> (min, max). Shared by data validation and the API.
RANGES: dict[str, tuple[float, float]] = {
    "limit_bal": (1, 1e7),
    "sex": (1, 2),
    "education": (0, 6),
    "marriage": (0, 3),
    "age": (18, 100),
    **{d: (-2, 9) for d in DELAYS},
    **{f"bill_amt{m}": (-1e7, 1e7) for m in MONTHS},
    **{f"pay_amt{m}": (0, 1e7) for m in MONTHS},
}
COLUMNS = list(RANGES)
FEATURES = [c for c in COLUMNS if c not in PROTECTED]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


Applicant: type[BaseModel] = create_model(
    "Applicant",
    __base__=_Strict,
    **{
        name: (Annotated[int if name in INTEGER else float, Field(ge=lo, le=hi)], ...)
        for name, (lo, hi) in RANGES.items()
        if name in FEATURES
    },
)
