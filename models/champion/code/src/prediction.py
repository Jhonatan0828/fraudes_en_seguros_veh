"""
Lógica de predicción y construcción del input para el modelo.
"""
from typing import Any, Dict, List

import pandas as pd

from .mappings import ORDINAL_MAPPINGS


def build_input_dataframe(form_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Construye el DataFrame con la estructura esperada por el pipeline.
    Aplica los mapeos ordinales y agrega los campos fijos.

    Args:
        form_data: diccionario con los valores ingresados por el usuario.

    Returns:
        DataFrame de una sola fila listo para predict_proba.
    """
    record = dict(form_data)

    # Aplicar mapeos ordinales
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in record and isinstance(record[col], str) and record[col] in mapping:
            record[col] = mapping[record[col]]

    # Campos fijos (valores medianos del dataset)
    record.setdefault("WeekOfMonth", 3)
    record.setdefault("WeekOfMonthClaimed", 3)
    record.setdefault("RepNumber", 12)

    return pd.DataFrame([record])


def predict_with_all_models(
    models: Dict[str, Any],
    X_input: pd.DataFrame,
    best_model_name: str,
) -> List[Dict[str, Any]]:
    """
    Genera predicciones con todos los modelos para una misma entrada.
    Útil para mostrar comparativa al usuario.
    """
    results = []
    for name, model in models.items():
        prob = model.predict_proba(X_input)[0, 1]
        results.append({
            "Modelo": name + (" ⭐" if name == best_model_name else ""),
            "Probabilidad fraude (%)": round(prob * 100, 2),
            "Predicción": "FRAUDE" if prob >= 0.5 else "Legítima",
        })
    return results


def detect_risk_factors(form_data: Dict[str, Any]) -> List[str]:
    """Detecta factores de riesgo legibles para el usuario."""
    factors = []

    if form_data.get("Fault") == "Policy Holder":
        factors.append("El asegurado es el responsable del siniestro")

    if form_data.get("BasePolicy") in ["Collision", "All Perils"]:
        factors.append(
            f"Póliza tipo {form_data['BasePolicy']} con alta incidencia de fraude"
        )

    if form_data.get("PastNumberOfClaims") in ["2 to 4", "more than 4"]:
        factors.append("Historial de múltiples reclamaciones previas")

    if form_data.get("AddressChange_Claim") in ["under 6 months", "1 year"]:
        factors.append("Cambio de dirección reciente antes de la reclamación")

    return factors
