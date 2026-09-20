"""
Adquisición de datos.

Obtiene el dataset crudo desde el origen configurado (archivo local o URL) y
valida que cumpla el contrato de datos esperado antes de dejarlo entrar al
pipeline. Si la validación falla, el flujo se detiene aquí en vez de entrenar
un modelo sobre datos corruptos.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..config import (
    RAW_DATASET, RAW_REQUIRED_COLUMNS, SENTINEL_CATEGORICAL_COLUMNS,
    SENTINEL_VALUE, TARGET_COLUMN,
)


def download_dataset(url: str, destination: Path) -> Path:
    """Descarga el CSV crudo desde una URL y lo cachea en disco."""
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    with open(destination, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 16):
            handle.write(chunk)
    return destination


def acquire_dataset(
    source: str = "local",
    url: Optional[str] = None,
    destination: Path = RAW_DATASET,
) -> pd.DataFrame:
    """
    Carga el dataset crudo desde el origen indicado.

    Args:
        source: "local" para leer el archivo ya presente, "url" para descargarlo.
        url: dirección del CSV cuando `source="url"`.
        destination: ruta donde vive (o se cachea) el archivo crudo.
    """
    destination = Path(destination)

    if source == "url":
        if not url:
            raise ValueError("Se requiere una URL cuando source='url'.")
        download_dataset(url, destination)
    elif source != "local":
        raise ValueError(f"Origen de datos no soportado: {source!r}")

    if not destination.exists():
        raise FileNotFoundError(
            f"Dataset crudo no encontrado en {destination}. "
            "Copia 'fraud_oracle.csv' en data/raw/ o ejecuta el flow con source='url'."
        )

    # utf-8-sig elimina el BOM con el que viene el CSV original.
    return pd.read_csv(destination, encoding="utf-8-sig")


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verifica el contrato de datos y devuelve un reporte de calidad.

    Raises:
        ValueError: si faltan columnas obligatorias, no hay filas o el target
            no es binario.
    """
    missing = sorted(set(RAW_REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"El dataset no trae las columnas obligatorias: {missing}")

    if df.empty:
        raise ValueError("El dataset crudo está vacío.")

    target_values = set(df[TARGET_COLUMN].dropna().unique())
    if not target_values <= {0, 1}:
        raise ValueError(
            f"La variable objetivo debe ser binaria (0/1), se encontró: {sorted(target_values)}"
        )

    sentinels = {
        column: int((df[column] == SENTINEL_VALUE).sum())
        for column in SENTINEL_CATEGORICAL_COLUMNS
        if column in df.columns
    }
    sentinels["Age"] = int((df["Age"] == 0).sum())

    fraud_rate = float(df[TARGET_COLUMN].mean())
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "duplicated_rows": int(df.duplicated().sum()),
        "null_values": int(df.isnull().sum().sum()),
        "sentinel_values": sentinels,
        "fraud_rate": round(fraud_rate, 6),
        "class_balance_ratio": round((1 - fraud_rate) / fraud_rate, 2),
    }


def save_quality_report(report: Dict[str, Any], destination: Path) -> Path:
    """Persiste el reporte de calidad para adjuntarlo como artefacto en MLflow."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination
