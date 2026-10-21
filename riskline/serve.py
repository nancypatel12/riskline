"""FastAPI service: validated inputs, versioned model, latency header."""

import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from riskline import __version__
from riskline.registry import Registry
from riskline.schema import FEATURES, Applicant

MAX_BATCH = 1000


class Prediction(BaseModel):
    probability: float = Field(description="Calibrated probability of default next month.")
    high_risk: bool = Field(description="probability >= the cost-optimal threshold.")


class PredictResponse(Prediction):
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    threshold: float
    model_version: str


def create_app(registry: str | None = None, version: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        root = registry or os.getenv("RISKLINE_REGISTRY", "models")
        app.state.model, app.state.card, _ = Registry(root).load(version or os.getenv("RISKLINE_MODEL_VERSION"))
        yield

    app = FastAPI(title="riskline", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def latency(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Latency-Ms"] = f"{(time.perf_counter() - start) * 1e3:.3f}"
        return response

    def score(rows: list) -> tuple[list[Prediction], dict]:
        card = app.state.card
        X = pd.DataFrame([r.model_dump() for r in rows], columns=FEATURES)
        p = app.state.model.predict_proba(X)[:, 1]
        return [Prediction(probability=float(x), high_risk=bool(x >= card["threshold"])) for x in p], card

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": app.state.card["version"]}

    @app.get("/model")
    def model() -> dict:
        card = app.state.card
        keys = ("version", "created_at", "threshold", "cost", "metrics", "features", "sklearn")
        return {k: card[k] for k in keys}

    @app.post("/predict")
    def predict(applicant: Applicant) -> PredictResponse:
        [pred], card = score([applicant])
        return PredictResponse(**pred.model_dump(), threshold=card["threshold"], model_version=card["version"])

    @app.post("/predict/batch")
    def predict_batch(batch: BatchRequest) -> BatchResponse:
        preds, card = score(batch.applicants)
        return BatchResponse(predictions=preds, threshold=card["threshold"], model_version=card["version"])

    return app


app = create_app()
