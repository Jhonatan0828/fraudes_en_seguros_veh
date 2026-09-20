"""
Registro y versionado con MLflow.

Cada modelo entrenado queda como un run con sus parámetros, métricas, firma y
artefactos. El candidato ganador se registra en el Model Registry y recibe el
alias `champion`, que es la única referencia que usa el servicio de predicción.
Así la promoción de un modelo no requiere tocar el código de la API.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from ..config import (
    CHAMPION_ALIAS, CHAMPION_EXPORT_DIR, CHAMPION_METADATA_FILE,
    MIN_SELECTION_METRIC, MLFLOW_ARTIFACT_LOCATION, MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI, MODEL_URI, REGISTERED_MODEL_NAME, ROOT_DIR,
    SELECTION_METRIC,
)


def model_requirements() -> List[str]:
    """
    Dependencias mínimas para deserializar y ejecutar el modelo.

    Se declaran explícitamente porque la inferencia automática de MLflow toma
    el entorno completo del proyecto (Jupyter, Streamlit, versiones en
    conflicto) y esa lista es la que instala la imagen Docker.
    `python-dotenv` entra porque el código del pipeline que viaja con el
    modelo importa la configuración del proyecto.
    """
    import cloudpickle
    import imblearn
    import numpy
    import pandas
    import scipy
    import sklearn
    import xgboost

    return [
        f"mlflow=={mlflow.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"xgboost=={xgboost.__version__}",
        f"imbalanced-learn=={imblearn.__version__}",
        f"pandas=={pandas.__version__}",
        f"numpy=={numpy.__version__}",
        f"scipy=={scipy.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
        "python-dotenv>=1.0",
    ]


def setup_mlflow() -> None:
    """Apunta MLflow al backend configurado y asegura el experimento."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MlflowClient().get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_tuning_run(tuning_results: Dict[str, Any], tags: Optional[Dict[str, str]] = None) -> str:
    """
    Registra la búsqueda de hiperparámetros: un run padre con el resumen y un
    run anidado por cada combinación evaluada.
    """
    setup_mlflow()
    with mlflow.start_run(run_name="hyperparameter-search", tags=tags) as parent:
        mlflow.set_tag("stage", "tuning")
        mlflow.log_params({
            "search_n_iter": tuning_results["n_iter"],
            "search_cv_folds": tuning_results["cv_folds"],
            "search_scoring": tuning_results["scoring"],
        })
        mlflow.log_metric("best_cv_pr_auc", tuning_results["best_score"])
        mlflow.log_params({
            f"best_{key}": value for key, value in tuning_results["best_params"].items()
        })

        for candidate in tuning_results["candidates"]:
            with mlflow.start_run(run_name=f"candidate-{candidate['rank']:02d}", nested=True):
                mlflow.log_params(candidate["params"])
                mlflow.log_metric("cv_pr_auc", candidate["mean_test_score"])
                mlflow.log_metric("cv_pr_auc_std", candidate["std_test_score"])

        return parent.info.run_id


def log_model_run(
    model_name: str,
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    X_sample: pd.DataFrame,
    training_seconds: float,
    artifacts: Optional[List[Path]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Registra un modelo entrenado como un run de MLflow.

    El modelo se guarda con firma y con el código de `src/`, de modo que pueda
    deserializarse en un entorno donde el repositorio no esté instalado.
    """
    setup_mlflow()
    with mlflow.start_run(run_name=model_name, tags=tags) as run:
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("model_name", model_name)
        mlflow.log_params(params)
        mlflow.log_metric("training_seconds", training_seconds)
        mlflow.log_metrics({
            key: value for key, value in metrics.items() if isinstance(value, (int, float))
        })

        for artifact in artifacts or []:
            mlflow.log_artifact(str(artifact))

        signature = infer_signature(X_sample, model.predict_proba(X_sample)[:, 1])
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_sample.head(3),
            code_paths=[str(ROOT_DIR / "src")],
            pip_requirements=model_requirements(),
            # El formato skops (por defecto en MLflow 3) rechaza los
            # transformadores propios del pipeline de limpieza.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "metrics": metrics,
        }


def select_candidate(
    runs: List[Dict[str, Any]],
    metric: str = SELECTION_METRIC,
    minimum: float = MIN_SELECTION_METRIC,
) -> Dict[str, Any]:
    """
    Elige el modelo candidato a producción por la métrica de decisión.

    Raises:
        ValueError: si el mejor modelo no alcanza el mínimo exigido. Es la
            compuerta de calidad que evita promover un modelo deficiente.
    """
    if not runs:
        raise ValueError("No hay modelos entrenados para comparar.")

    best = max(runs, key=lambda run: run["metrics"][metric])
    score = best["metrics"][metric]
    if score < minimum:
        raise ValueError(
            f"Ningún modelo alcanza el mínimo de {metric}={minimum}. "
            f"El mejor fue {best['model_name']} con {score:.4f}."
        )
    return best


def register_champion(candidate: Dict[str, Any], tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Registra el candidato en el Model Registry y le asigna el alias `champion`.

    El umbral de decisión viaja como tag de la versión para que el servicio de
    predicción lo lea sin necesidad de recalcularlo.
    """
    setup_mlflow()
    client = MlflowClient()

    version = mlflow.register_model(
        model_uri=candidate["model_uri"],
        name=REGISTERED_MODEL_NAME,
        tags=tags,
    )
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)

    metrics = candidate["metrics"]
    version_tags = {
        "model_name": candidate["model_name"],
        "decision_threshold": f"{metrics['decision_threshold']:.6f}",
        "selection_metric": SELECTION_METRIC,
        "selection_score": f"{metrics[SELECTION_METRIC]:.6f}",
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, key, value)

    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "version": int(version.version),
        "alias": CHAMPION_ALIAS,
        "run_id": candidate["run_id"],
        "model_name": candidate["model_name"],
    }


def export_champion(destination: Path = CHAMPION_EXPORT_DIR) -> Path:
    """
    Descarga el modelo con alias `champion` a una carpeta local.

    Este export es el artefacto que empaqueta la imagen Docker: permite servir
    el modelo sin depender de que el servidor de MLflow esté disponible.
    """
    setup_mlflow()
    client = MlflowClient()
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)

    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}",
            dst_path=tmp,
        )
        shutil.copytree(downloaded, destination)

    metadata = {
        "registered_model": REGISTERED_MODEL_NAME,
        "version": int(version.version),
        "run_id": version.run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **version.tags,
    }
    (destination / CHAMPION_METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination


def load_champion(uri: str = MODEL_URI) -> Tuple[Any, Dict[str, Any]]:
    """
    Carga el modelo de producción y su metadata.

    Acepta tanto una carpeta local exportada como una referencia al registro
    (`models:/fraud-detection-classifier@champion`).
    """
    if uri.startswith(("models:", "runs:")):
        setup_mlflow()
        model = mlflow.sklearn.load_model(uri)
        client = MlflowClient()
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        metadata = {
            "registered_model": REGISTERED_MODEL_NAME,
            "version": int(version.version),
            "run_id": version.run_id,
            **version.tags,
        }
        return model, metadata

    model_dir = Path(uri)
    model = mlflow.sklearn.load_model(str(model_dir))
    metadata_file = model_dir / CHAMPION_METADATA_FILE
    metadata = (
        json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata_file.exists()
        else {"registered_model": REGISTERED_MODEL_NAME, "source": str(model_dir)}
    )
    return model, metadata
