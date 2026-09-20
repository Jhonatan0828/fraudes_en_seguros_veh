"""
Procesamiento de datos: transformaciones derivadas del EDA.

Las tres transformaciones se implementan como estimadores de scikit-learn para
que viajen dentro del modelo serializado. Así el servicio de predicción aplica
exactamente las mismas reglas que se usaron en el entrenamiento, sin duplicar
lógica en la API.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from ..config import COLUMNS_TO_DROP, SENTINEL_CATEGORICAL_COLUMNS, SENTINEL_VALUE
from ..mappings import ORDINAL_MAPPINGS


class SentinelImputer(BaseEstimator, TransformerMixin):
    """
    Imputa los faltantes encubiertos detectados en el EDA.

    En este dataset los faltantes no son nulos: `Age == 0` (319 registros) y
    `MonthClaimed`/`DayOfWeekClaimed == '0'`. Los estadísticos de imputación se
    aprenden únicamente en `fit`, por lo que al usarse dentro del pipeline se
    calculan solo con el set de entrenamiento y no filtran información del test.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        sentinel: str = SENTINEL_VALUE,
    ):
        self.categorical_columns = categorical_columns
        self.sentinel = sentinel

    def _categorical_columns(self) -> List[str]:
        if self.categorical_columns is None:
            return list(SENTINEL_CATEGORICAL_COLUMNS)
        return list(self.categorical_columns)

    def fit(self, X: pd.DataFrame, y=None) -> "SentinelImputer":
        self.age_median_ = (
            int(X.loc[X["Age"] > 0, "Age"].median()) if "Age" in X.columns else None
        )
        self.category_modes_: Dict[str, str] = {}
        for column in self._categorical_columns():
            if column not in X.columns:
                continue
            valid = X.loc[X[column] != self.sentinel, column]
            if not valid.empty:
                self.category_modes_[column] = valid.mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "Age" in X.columns and self.age_median_ is not None:
            X.loc[X["Age"] == 0, "Age"] = self.age_median_
        for column, mode in self.category_modes_.items():
            if column in X.columns:
                X.loc[X[column] == self.sentinel, column] = mode
        return X


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """
    Convierte a numéricas las variables ordinales expresadas como texto
    (p. ej. `VehiclePrice`: 'less than 20000' → 0, 'more than 69000' → 5).

    Los valores que ya son numéricos se conservan tal cual, de modo que la misma
    transformación sirve para el CSV crudo y para registros que ya vienen
    codificados (como los que envía la app de Streamlit).
    """

    def __init__(self, mappings: Optional[Dict[str, Dict[str, int]]] = None):
        self.mappings = mappings

    def _mappings(self) -> Dict[str, Dict[str, int]]:
        return ORDINAL_MAPPINGS if self.mappings is None else self.mappings

    def fit(self, X: pd.DataFrame, y=None) -> "OrdinalMapper":
        self.mappings_ = self._mappings()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column, mapping in self.mappings_.items():
            if column not in X.columns:
                continue
            mapped = X[column].map(mapping)
            already_numeric = pd.to_numeric(X[column], errors="coerce")
            resolved = mapped.where(mapped.notna(), already_numeric)
            # El downcast deja el mismo dtype entero venga el valor como
            # etiqueta o ya codificado; si hay categorías desconocidas quedan
            # como NaN y las imputa el ColumnTransformer.
            X[column] = pd.to_numeric(resolved, downcast="integer")
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Elimina las columnas sin poder predictivo: el identificador de la póliza,
    el año (no se repite en datos futuros) y el rango etario del asegurado
    (redundante con `Age`).
    """

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns

    def _columns(self) -> List[str]:
        return list(COLUMNS_TO_DROP) if self.columns is None else list(self.columns)

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnDropper":
        self.columns_ = self._columns()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        present = [column for column in self.columns_ if column in X.columns]
        return X.drop(columns=present)


def cleaning_steps() -> List[Tuple[str, BaseEstimator]]:
    """
    Pasos de limpieza en el orden correcto, como lista plana.

    Se exponen sueltos porque los pipelines de imbalanced-learn no admiten
    pipelines anidados como pasos intermedios.
    """
    return [
        ("sentinels", SentinelImputer()),
        ("ordinal", OrdinalMapper()),
        ("dropper", ColumnDropper()),
    ]


def build_cleaning_pipeline() -> Pipeline:
    """Encadena las transformaciones de limpieza en un pipeline reutilizable."""
    return Pipeline(cleaning_steps())
