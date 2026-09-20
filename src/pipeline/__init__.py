"""
Pipeline de Machine Learning para detección de fraude en seguros vehiculares.

Cada módulo corresponde a una etapa del ciclo de vida que orquesta Prefect:

    ingestion     → adquisición y validación de los datos crudos
    preprocessing → transformaciones derivadas del EDA (limpieza)
    features      → feature engineering y partición train/test
    tuning        → optimización de hiperparámetros
    training      → definición y entrenamiento de los modelos candidatos
    evaluation    → métricas de desempeño y selección de umbral
    registry      → registro, versionado y promoción en MLflow
"""
