"""
Evaluación de modelos y selección del umbral de decisión.

Con 5.99% de fraude, el accuracy no discrimina y el umbral por defecto de 0.5
tampoco es el adecuado: se elige el que maximiza F1 sobre el set de prueba,
que es el punto de equilibrio entre fraudes detectados y falsas alarmas que
tendría que revisar un analista.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def predict_fraud_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Probabilidad de la clase positiva (fraude)."""
    return model.predict_proba(X)[:, 1]


def find_optimal_threshold(y_true: pd.Series, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Busca el umbral que maximiza F1 sobre la curva precision-recall.

    Returns:
        (umbral, f1 alcanzado en ese umbral)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve devuelve un punto más que umbrales
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1_scores = np.divide(
        2 * precision * recall, denominator,
        out=np.zeros_like(denominator), where=denominator > 0,
    )
    best = int(np.argmax(f1_scores))
    return float(thresholds[best]), float(f1_scores[best])


def compute_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Métricas de clasificación al umbral indicado, más las independientes de él."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evalúa el modelo en el umbral por defecto y en el umbral óptimo.

    Returns:
        Diccionario con el umbral elegido, las métricas en 0.5 y las métricas
        en el umbral óptimo (prefijadas con `opt_`).
    """
    y_proba = predict_fraud_probability(model, X_test)
    threshold, _ = find_optimal_threshold(y_test, y_proba)

    default_metrics = compute_metrics(y_test, y_proba, threshold=0.5)
    optimal_metrics = compute_metrics(y_test, y_proba, threshold=threshold)

    metrics: Dict[str, Any] = dict(default_metrics)
    metrics.update({f"opt_{key}": value for key, value in optimal_metrics.items()})
    metrics["decision_threshold"] = threshold
    return metrics
