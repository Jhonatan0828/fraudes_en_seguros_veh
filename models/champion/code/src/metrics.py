"""
Cálculo de métricas y curvas para evaluación de modelos.
"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)


def compute_classification_metrics(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Calcula métricas de clasificación y curvas para todos los modelos.

    Returns:
        - DataFrame con métricas (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
        - Diccionario con curvas ROC, PR y matriz de confusión por modelo
    """
    rows = []
    curves = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        rows.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "PR-AUC": average_precision_score(y_test, y_proba),
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision_curve, recall_curve_, _ = precision_recall_curve(y_test, y_proba)

        curves[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision_curve,
            "recall": recall_curve_,
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "cm": confusion_matrix(y_test, y_pred),
        }

    return pd.DataFrame(rows), curves


def get_best_model_name(metrics_df: pd.DataFrame, by: str = "PR-AUC") -> str:
    """Devuelve el nombre del mejor modelo según la métrica especificada."""
    return metrics_df.loc[metrics_df[by].idxmax(), "Modelo"]
