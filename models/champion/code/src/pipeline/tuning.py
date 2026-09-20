"""
Optimización de hiperparámetros.

Búsqueda aleatoria sobre XGBoost —el candidato más fuerte de la entrega
anterior— optimizando PR-AUC con validación cruzada estratificada. La búsqueda
se hace solo sobre el set de entrenamiento: el de prueba queda intacto para la
evaluación final.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from ..config import RANDOM_STATE, TUNING_CV_FOLDS, TUNING_N_ITER
from .training import build_model_pipelines

SEARCH_SPACE: Dict[str, List[Any]] = {
    "clf__n_estimators": [200, 300, 400, 600],
    "clf__max_depth": [3, 4, 6, 8],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
    "clf__min_child_weight": [1, 3, 5],
    "clf__gamma": [0, 0.5, 1.0],
}


def optimize_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    n_iter: int = TUNING_N_ITER,
    cv_folds: int = TUNING_CV_FOLDS,
) -> Dict[str, Any]:
    """
    Ejecuta la búsqueda aleatoria y devuelve el resultado completo.

    Returns:
        best_params: hiperparámetros ganadores, sin el prefijo `clf__`.
        best_score:  PR-AUC medio en validación cruzada.
        candidates:  cada combinación probada, para registrarla en MLflow.
    """
    estimator = build_model_pipelines(scale_pos_weight)["XGBoost"]
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=SEARCH_SPACE,
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=False,
        verbose=0,
    )
    search.fit(X_train, y_train)

    results = search.cv_results_
    candidates = [
        {
            "params": {key.removeprefix("clf__"): value for key, value in params.items()},
            "mean_test_score": float(results["mean_test_score"][index]),
            "std_test_score": float(results["std_test_score"][index]),
            "rank": int(results["rank_test_score"][index]),
        }
        for index, params in enumerate(results["params"])
    ]

    return {
        "best_params": {
            key.removeprefix("clf__"): value for key, value in search.best_params_.items()
        },
        "best_score": float(search.best_score_),
        "candidates": sorted(candidates, key=lambda candidate: candidate["rank"]),
        "n_iter": n_iter,
        "cv_folds": cv_folds,
        "scoring": "average_precision",
    }
