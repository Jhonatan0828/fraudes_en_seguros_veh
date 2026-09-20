"""
Flujo de entrenamiento orquestado con Prefect.

Encadena el ciclo de vida completo:

    adquisición → validación → partición → feature engineering →
    optimización → entrenamiento → evaluación → selección → registro → export

Ejecución local:
    python -m flows.training_flow
    python -m flows.training_flow --no-tuning        # más rápido, sin búsqueda
    python -m flows.training_flow --source url --url https://...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.runtime import flow_run

from src.config import (
    CLEAN_DATASET, COLUMNS_TO_DROP, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
    REPORTS_DIR, SELECTION_METRIC, TARGET_COLUMN, TUNING_ENABLED,
)
from src.pipeline.evaluation import evaluate_model
from src.pipeline.features import (
    compute_scale_pos_weight, split_features_target, stratified_split,
)
from src.pipeline.ingestion import acquire_dataset, save_quality_report, validate_dataset
from src.pipeline.preprocessing import build_cleaning_pipeline
from src.pipeline.registry import (
    export_champion, log_model_run, log_tuning_run, register_champion, select_candidate,
)
from src.pipeline.training import build_model_pipelines, classifier_params, train_model
from src.pipeline.tuning import optimize_xgboost

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "Logistic + SMOTE"]


@task(name="adquirir-datos", retries=2, retry_delay_seconds=10)
def acquire_data(source: str, url: Optional[str]) -> pd.DataFrame:
    """Etapa 1 — Adquisición: trae el dataset crudo desde el origen configurado."""
    logger = get_run_logger()
    df = acquire_dataset(source=source, url=url)
    logger.info("Dataset adquirido desde '%s': %s filas x %s columnas",
                source, len(df), df.shape[1])
    return df


@task(name="validar-datos")
def validate_data(df: pd.DataFrame) -> Path:
    """Etapa 2 — Validación: verifica el contrato de datos y guarda el reporte."""
    logger = get_run_logger()
    report = validate_dataset(df)
    path = save_quality_report(report, REPORTS_DIR / "data_quality_report.json")
    logger.info("Calidad de datos: %s filas, %s duplicados, tasa de fraude %.4f",
                report["n_rows"], report["duplicated_rows"], report["fraud_rate"])
    return path


@task(name="particionar-datos")
def split_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Etapa 3 — Partición estratificada previa a cualquier ajuste."""
    logger = get_run_logger()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    logger.info("Train: %s | Test: %s | fraude train %.4f, test %.4f",
                X_train.shape, X_test.shape, y_train.mean(), y_test.mean())
    return X_train, X_test, y_train, y_test


@task(name="persistir-dataset-procesado")
def persist_processed_dataset(df: pd.DataFrame, X_train: pd.DataFrame) -> Path:
    """
    Etapa 4 — Procesamiento: aplica la limpieza al dataset completo y lo guarda.

    Las reglas de imputación se aprenden solo con el set de entrenamiento; el
    CSV resultante alimenta el EDA y la app de Streamlit.
    """
    logger = get_run_logger()
    X, y = split_features_target(df)
    cleaning = build_cleaning_pipeline().fit(X_train)
    processed = cleaning.transform(X)
    processed[TARGET_COLUMN] = y.values
    # Se conserva el orden de columnas del dataset original
    processed = processed[[c for c in df.columns if c not in COLUMNS_TO_DROP]]

    CLEAN_DATASET.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(CLEAN_DATASET, index=False)
    logger.info("Dataset procesado guardado en %s (%s columnas)",
                CLEAN_DATASET, processed.shape[1])
    return CLEAN_DATASET


@task(name="optimizar-hiperparametros")
def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    tags: Dict[str, str],
) -> Dict[str, Any]:
    """Etapa 5 — Optimización: búsqueda aleatoria de hiperparámetros para XGBoost."""
    logger = get_run_logger()
    results = optimize_xgboost(X_train, y_train, scale_pos_weight)
    log_tuning_run(results, tags=tags)
    logger.info("Mejor PR-AUC en validación cruzada: %.4f con %s",
                results["best_score"], results["best_params"])
    return results


@task(name="entrenar-y-registrar-modelo")
def train_and_log(
    model_name: str,
    scale_pos_weight: float,
    xgb_params: Optional[Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    quality_report: Path,
    tags: Dict[str, str],
) -> Dict[str, Any]:
    """Etapa 6 — Entrenamiento y evaluación de un candidato, registrado en MLflow."""
    logger = get_run_logger()
    model = build_model_pipelines(scale_pos_weight, xgb_params)[model_name]
    model, seconds = train_model(model, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    result = log_model_run(
        model_name=model_name,
        model=model,
        params=classifier_params(model),
        metrics=metrics,
        X_sample=X_test,
        training_seconds=seconds,
        artifacts=[quality_report],
        tags=tags,
    )
    logger.info("%s → PR-AUC %.4f | ROC-AUC %.4f | F1(opt) %.4f | %.1fs",
                model_name, metrics["pr_auc"], metrics["roc_auc"],
                metrics["opt_f1"], seconds)
    return result


@task(name="seleccionar-candidato")
def select_production_candidate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Etapa 7 — Selección del modelo candidato a producción."""
    logger = get_run_logger()
    candidate = select_candidate(runs)
    logger.info("Candidato a producción: %s (%s=%.4f)", candidate["model_name"],
                SELECTION_METRIC, candidate["metrics"][SELECTION_METRIC])
    return candidate


@task(name="registrar-campeon")
def promote_champion(candidate: Dict[str, Any], tags: Dict[str, str]) -> Dict[str, Any]:
    """Etapa 8 — Registro y versionado: promueve el candidato a `champion`."""
    logger = get_run_logger()
    registration = register_champion(candidate, tags=tags)
    export_champion()
    logger.info("Registrado %s v%s con alias '%s'", registration["registered_model"],
                registration["version"], registration["alias"])
    return registration


@task(name="reporte-comparativo")
def write_comparison_report(
    runs: List[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> Path:
    """Genera la tabla comparativa de modelos para la sustentación."""
    columns = ["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy",
               "decision_threshold", "opt_f1", "opt_recall", "opt_precision"]
    table = pd.DataFrame([
        {"Modelo": run["model_name"], **{c: run["metrics"][c] for c in columns}}
        for run in runs
    ]).sort_values("pr_auc", ascending=False).round(4)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    markdown = REPORTS_DIR / "model_comparison.md"
    markdown.write_text(
        "# Comparación de modelos candidatos\n\n"
        f"Métrica de decisión: **{SELECTION_METRIC}** "
        "(PR-AUC, apropiada con 5.99% de clase positiva).\n\n"
        f"{table.to_markdown(index=False)}\n\n"
        f"## Modelo candidato\n\n**{candidate['model_name']}** — "
        f"{SELECTION_METRIC} = {candidate['metrics'][SELECTION_METRIC]:.4f}, "
        f"umbral de decisión = {candidate['metrics']['decision_threshold']:.4f}\n",
        encoding="utf-8",
    )
    return markdown


@flow(name="fraude-pipeline-entrenamiento")
def training_flow(
    source: str = "local",
    url: Optional[str] = None,
    tuning: bool = TUNING_ENABLED,
    promote: bool = True,
) -> Dict[str, Any]:
    """
    Orquesta el ciclo de vida completo del modelo de detección de fraude.

    Args:
        source: "local" o "url" para la etapa de adquisición.
        url: dirección del CSV cuando `source="url"`.
        tuning: ejecuta la búsqueda de hiperparámetros antes de entrenar.
        promote: promueve el candidato ganador en el Model Registry.
    """
    logger = get_run_logger()
    logger.info("MLflow tracking: %s | experimento: %s",
                MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)

    tags = {"prefect_flow_run_id": str(flow_run.id), "dataset": "fraud_oracle"}

    raw = acquire_data(source, url)
    quality_report = validate_data(raw)
    X_train, X_test, y_train, y_test = split_dataset(raw)
    persist_processed_dataset(raw, X_train)

    scale_pos_weight = compute_scale_pos_weight(y_train)

    xgb_params = None
    if tuning:
        xgb_params = optimize_hyperparameters(
            X_train, y_train, scale_pos_weight, tags
        )["best_params"]

    runs = [
        train_and_log(
            model_name, scale_pos_weight, xgb_params,
            X_train, y_train, X_test, y_test, quality_report, tags,
        )
        for model_name in MODEL_NAMES
    ]

    candidate = select_production_candidate(runs)
    report = write_comparison_report(runs, candidate)

    summary: Dict[str, Any] = {
        "candidate": candidate["model_name"],
        "metrics": candidate["metrics"],
        "report": str(report),
    }
    if promote:
        summary["registration"] = promote_champion(candidate, tags)

    logger.info("Pipeline finalizado. Candidato: %s", summary["candidate"])
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento (Prefect + MLflow)")
    parser.add_argument("--source", default="local", choices=["local", "url"])
    parser.add_argument("--url", default=None, help="URL del CSV crudo si --source url")
    parser.add_argument("--no-tuning", action="store_true", help="Omite la búsqueda de hiperparámetros")
    parser.add_argument("--no-promote", action="store_true", help="No promueve el modelo en el registro")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    training_flow(
        source=args.source,
        url=args.url,
        tuning=not args.no_tuning,
        promote=not args.no_promote,
    )
