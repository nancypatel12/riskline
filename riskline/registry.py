"""A file-based model registry with a champion/challenger gate."""

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn

GATE_METRIC = "pr_auc"


class Registry:
    def __init__(self, root: str | Path = "models"):
        self.root = Path(root)
        self.index = self.root / "registry.json"

    def _read(self) -> dict:
        if self.index.exists():
            return json.loads(self.index.read_text())
        return {"production": None, "versions": {}}

    def register(self, model, card: dict, reference: pd.DataFrame) -> tuple[str, bool]:
        """Store a new version; promote it if it beats production. Returns (version, promoted)."""
        index = self._read()
        version = f"v{len(index['versions']) + 1}"
        path = self.root / version
        path.mkdir(parents=True, exist_ok=False)

        card = {
            "version": version,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            **card,
        }
        joblib.dump(model, path / "model.joblib")
        reference.to_parquet(path / "reference.parquet", index=False)
        (path / "card.json").write_text(json.dumps(card, indent=2))

        champion = index["production"]
        score = card["metrics"][GATE_METRIC]
        best = index["versions"][champion][GATE_METRIC] if champion else -1.0
        promoted = score >= best
        index["versions"][version] = {GATE_METRIC: score, "created_at": card["created_at"]}
        if promoted:
            index["production"] = version
        self.index.write_text(json.dumps(index, indent=2))
        return version, promoted

    def load(self, version: str | None = None):
        """Return (model, card, reference) for a version, defaulting to production."""
        version = version or self._read()["production"]
        if version is None:
            raise FileNotFoundError(f"no production model in {self.root}; run `make train`")
        path = self.root / version
        card = json.loads((path / "card.json").read_text())
        return (
            joblib.load(path / "model.joblib"),
            card,
            pd.read_parquet(path / "reference.parquet"),
        )
