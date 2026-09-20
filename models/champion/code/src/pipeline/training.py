"""
Entrenamiento de los modelos candidatos.

Cada candidato es un pipeline completo: limpieza → feature engineering →
clasificador. Al estar todo dentro del mismo objeto, el modelo registrado en
MLflow recibe registros crudos y devuelve una probabilidad, sin pasos manuales.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ..config import RANDOM_STATE
from .features import build_preprocessor
from .preprocessing import cleaning_steps

DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
}


def build_model_pipelines(
    scale_pos_weight: float,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Pipeline]:
    """
    Define los cuatro candidatos que compiten por salir a producción.

    Args:
        scale_pos_weight: peso de la clase minoritaria para XGBoost.
        xgb_params: hiperparámetros de XGBoost; si es None se usan los de base.
    """
    xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    logistic = Pipeline([
        *cleaning_steps(),
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=2000,
            solver="lbfgs", random_state=RANDOM_STATE,
        )),
    ])

    random_forest = Pipeline([
        *cleaning_steps(),
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
        )),
    ])

    xgboost = Pipeline([
        *cleaning_steps(),
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE, **xgb_params,
        )),
    ])

    logistic_smote = ImbPipeline([
        *cleaning_steps(),
        ("prep", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE,
        )),
    ])

    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
        "Logistic + SMOTE": logistic_smote,
    }


def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Pipeline, float]:
    """Entrena el pipeline y devuelve el modelo ajustado y su duración."""
    started = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - started


def classifier_params(model: Pipeline) -> Dict[str, Any]:
    """Extrae los hiperparámetros del clasificador para registrarlos en MLflow."""
    classifier = model.named_steps["clf"]
    return {
        key: value
        for key, value in classifier.get_params(deep=False).items()
        if value is not None and not callable(value)
    }
