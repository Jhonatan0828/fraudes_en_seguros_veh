"""
Feature engineering y partición de los datos.

El `ColumnTransformer` escala las variables numéricas/ordinales y aplica
One-Hot Encoding a las nominales. Se construye con una función (no como objeto
compartido) para que cada modelo reciba su propia instancia sin estado heredado.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import (
    NOMINAL_COLUMNS, NUMERIC_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE,
)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa la matriz de features de la variable objetivo."""
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partición estratificada que preserva el 5.99% de fraude en ambos sets."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Construye el transformador de features.

    Los imputadores cubren valores desconocidos que puedan llegar desde la API
    (por ejemplo una categoría nueva que el mapeo ordinal no reconoce).
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, NOMINAL_COLUMNS),
        ],
        remainder="drop",
    )


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Razón negativos/positivos que compensa el desbalance en XGBoost."""
    positives = int((y == 1).sum())
    if positives == 0:
        raise ValueError("El set de entrenamiento no contiene casos de fraude.")
    return float((y == 0).sum() / positives)
