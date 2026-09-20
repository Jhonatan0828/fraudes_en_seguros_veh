"""
Servicio de predicción de fraude.

Carga el modelo con alias `champion` (desde el Model Registry o desde el export
local que empaqueta la imagen Docker) y expone la puntuación por REST.

Ejecución local:
    uvicorn src.api.main:app --reload
Documentación interactiva: http://localhost:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI

from ..config import DEFAULT_DECISION_THRESHOLD, MODEL_URI, REGISTERED_MODEL_NAME
from ..pipeline.registry import load_champion
from .schemas import (
    BatchClaimRequest, BatchPredictionResponse, ClaimRequest, HealthResponse,
    ModelInfoResponse, PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez, al arrancar el servicio."""
    model, metadata = load_champion(MODEL_URI)
    app.state.model = model
    app.state.metadata = metadata
    app.state.threshold = float(
        metadata.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
    )
    yield


app = FastAPI(
    title="API de Detección de Fraude en Seguros Vehiculares",
    description=(
        "Servicio de puntuación del modelo campeón registrado en MLflow. "
        "Universidad de Medellín — Aprendizaje Automático en la Nube."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _score(claims: List[ClaimRequest]) -> List[PredictionResponse]:
    """Puntúa un lote de reclamaciones con una sola llamada al modelo."""
    frame = pd.DataFrame([claim.model_dump() for claim in claims])
    probabilities = app.state.model.predict_proba(frame)[:, 1]
    threshold = app.state.threshold
    return [
        PredictionResponse(
            fraud_probability=round(float(probability), 6),
            is_fraud=bool(probability >= threshold),
            decision_threshold=threshold,
        )
        for probability in probabilities
    ]


@app.get("/health", response_model=HealthResponse, tags=["operación"])
def health() -> HealthResponse:
    """Estado del servicio y del modelo cargado."""
    return HealthResponse(
        status="ok",
        model_loaded=app.state.model is not None,
        registered_model=REGISTERED_MODEL_NAME,
        version=app.state.metadata.get("version"),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["operación"])
def model_info() -> ModelInfoResponse:
    """Versión, métrica de selección y umbral del modelo en producción."""
    metadata = app.state.metadata
    return ModelInfoResponse(
        registered_model=metadata.get("registered_model", REGISTERED_MODEL_NAME),
        version=metadata.get("version"),
        model_name=metadata.get("model_name"),
        run_id=metadata.get("run_id"),
        decision_threshold=app.state.threshold,
        selection_metric=metadata.get("selection_metric"),
        selection_score=metadata.get("selection_score"),
        promoted_at=metadata.get("promoted_at"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["predicción"])
def predict(claim: ClaimRequest) -> PredictionResponse:
    """Evalúa una reclamación individual."""
    return _score([claim])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["predicción"])
def predict_batch(request: BatchClaimRequest) -> BatchPredictionResponse:
    """Evalúa un lote de reclamaciones."""
    predictions = _score(request.claims)
    return BatchPredictionResponse(
        predictions=predictions,
        n_claims=len(predictions),
        n_flagged=sum(prediction.is_fraud for prediction in predictions),
    )
