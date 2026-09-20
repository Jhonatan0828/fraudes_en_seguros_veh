"""
Despliegue programado del flow de entrenamiento.

Registra el flow en el servidor de Prefect y lo deja escuchando con una
programación semanal. Mientras el proceso esté activo, Prefect dispara el
reentrenamiento y la promoción del campeón sin intervención manual.

    python -m flows.deployment

La interfaz de Prefect queda en http://localhost:4200 (`prefect server start`).
"""
from flows.training_flow import training_flow

if __name__ == "__main__":
    training_flow.serve(
        name="reentrenamiento-semanal",
        description=(
            "Reentrena los candidatos, compara por PR-AUC y promueve el campeón "
            "en el Model Registry de MLflow."
        ),
        cron="0 3 * * 1",  # lunes a las 3:00
        tags=["fraude", "entrenamiento", "mlflow"],
        parameters={"source": "local", "tuning": True, "register": True},
    )
