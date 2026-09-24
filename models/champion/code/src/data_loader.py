"""
Módulo de carga de datos y artefactos del modelo.
Centraliza la carga de archivos para evitar duplicación.
"""
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import pandas as pd

from .config import (
    BEST_MODEL_PATH, CLEAN_DATASET, METADATA_PATH, MODELS_DIR, RAW_DATASET,
)


def load_raw_dataset() -> pd.DataFrame:
    """Carga el dataset original directamente de la carpeta raw/."""
    if not RAW_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset original no encontrado en {RAW_DATASET}. "
            "Asegúrate de que el archivo CSV esté en la carpeta data/raw/."
        )
    return pd.read_csv(RAW_DATASET)

def load_clean_dataset() -> pd.DataFrame:
    """Carga el dataset preprocesado listo para el modelo."""
    if not CLEAN_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset limpio no encontrado en {CLEAN_DATASET}. "
            "Ejecuta el pipeline primero: python -m flows.training_flow"
        )
    return pd.read_csv(CLEAN_DATASET)


def load_metadata() -> Dict[str, Any]:
    """Carga el diccionario de metadata del entrenamiento."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata no encontrada en {METADATA_PATH}. "
            "La genera el notebook notebooks/fraud_detection_analysis.ipynb, "
            "que alimenta la app de exploración."
        )
    return joblib.load(METADATA_PATH)


def load_all_models(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Carga todos los modelos entrenados desde la carpeta models/.
    Si no encuentra la carpeta, devuelve solo el mejor modelo.
    """
    models_dict: Dict[str, Any] = {}

    if MODELS_DIR.is_dir():
        for name in metadata.get("all_model_names", []):
            safe_name = name.replace(" ", "_").replace("+", "plus")
            path = MODELS_DIR / f"{safe_name}.pkl"
            if path.exists():
                models_dict[name] = joblib.load(path)

    # Fallback: solo el mejor modelo
    if not models_dict:
        models_dict[metadata["best_model_name"]] = joblib.load(BEST_MODEL_PATH)

    return models_dict


def load_artifacts() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Carga modelos y metadata de una sola llamada."""
    metadata = load_metadata()
    models = load_all_models(metadata)
    return models, metadata
