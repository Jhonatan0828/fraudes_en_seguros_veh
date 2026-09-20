# 🔍 Detección de Fraude en Seguros Vehiculares — Pipeline de MLOps

> Proyecto académico — **Universidad de Medellín**
> Curso: Aprendizaje Automático en la Nube
> Profesora: Maria Camila Durango Barrera

[![Tests](https://github.com/Jhonatan0828/fraudes_en_seguros_veh/actions/workflows/tests.yml/badge.svg)](https://github.com/Jhonatan0828/fraudes_en_seguros_veh/actions)

## 📌 Descripción

Ciclo de vida completo de un modelo de detección de reclamaciones fraudulentas,
**orquestado con Prefect** y **versionado con MLflow**, desde la adquisición de
los datos hasta el despliegue del modelo como servicio REST en Docker.

Dataset: *Vehicle Insurance Claim Fraud Detection* (Oracle/Kaggle) — 15.420
reclamaciones del período 1994–1996, con 5,99% de fraude.

## 🏗️ Arquitectura del pipeline

```
                        ┌──────────────── Prefect (orquestación) ────────────────┐
                        │                                                        │
  data/raw/*.csv  ──►   │  1. adquirir-datos        →  validación de esquema      │
       │   URL          │  2. validar-datos         →  reporte de calidad         │
       ▼                │  3. particionar-datos     →  split estratificado 80/20  │
                        │  4. persistir-procesado   →  data/processed/*.csv       │
                        │  5. optimizar-hiperparám. →  RandomizedSearchCV         │
                        │  6. entrenar 4 candidatos →  ┐                          │
                        │  7. seleccionar candidato    │                          │
                        │  8. registrar campeón     →  ┘                          │
                        └───────────────────────────────┬────────────────────────┘
                                                        ▼
                                   MLflow: runs, métricas, firma, Model Registry
                                                        │  alias @champion
                                                        ▼
                                          models/champion/  (export)
                                                        │
                                                        ▼
                                    FastAPI + Docker  →  POST /predict
```

Cada modelo entrenado es un **pipeline completo de scikit-learn**
(limpieza → feature engineering → clasificador). El servicio recibe una
reclamación cruda y devuelve la probabilidad de fraude, sin transformaciones
manuales en el camino.

## 📁 Estructura del proyecto

```
├── flows/                          # Orquestación con Prefect
│   ├── training_flow.py            #   flow principal (9 tasks)
│   └── deployment.py               #   despliegue programado (cron semanal)
│
├── src/
│   ├── config.py                   # Configuración central y variables de entorno
│   ├── mappings.py                 # Mapeos ordinales (fuente única de verdad)
│   │
│   ├── pipeline/                   # Etapas del ciclo de vida
│   │   ├── ingestion.py            #   adquisición + contrato de datos
│   │   ├── preprocessing.py        #   transformadores de limpieza (sklearn)
│   │   ├── features.py             #   ColumnTransformer + partición
│   │   ├── tuning.py               #   optimización de hiperparámetros
│   │   ├── training.py             #   definición de los 4 candidatos
│   │   ├── evaluation.py           #   métricas + umbral de decisión
│   │   └── registry.py             #   MLflow: logging, registro y promoción
│   │
│   ├── api/                        # Servicio de predicción
│   │   ├── main.py                 #   endpoints FastAPI
│   │   └── schemas.py              #   contratos de entrada/salida
│   │
│   ├── data_loader.py              # Carga de artefactos para la app
│   ├── metrics.py, plotting.py     # Utilidades de la app
│   ├── prediction.py
│   └── pages/                      # Páginas de la app Streamlit
│
├── deployment/
│   ├── Dockerfile                  # Imagen del servicio de predicción
│   ├── docker-compose.yml
│   └── requirements-api.txt
│
├── notebooks/
│   └── fraud_detection_analysis.ipynb   # EDA y análisis exploratorio
│
├── data/
│   ├── raw/fraud_oracle.csv        # Dataset original (versionado)
│   └── processed/                  # Salida de la etapa de procesamiento
│
├── models/                         # Artefactos de modelos
│   └── champion/                   #   modelo promovido, exportado desde MLflow
│
├── reports/                        # Calidad de datos y comparación de modelos
├── tests/                          # Tests del pipeline, de la API y de la app
├── .github/workflows/tests.yml     # CI: pytest en cada push y pull request
│
├── app.py                          # App Streamlit (EDA interactivo)
├── pyproject.toml                  # Dependencias y configuración del paquete
├── requirements.txt                # Versiones fijadas para despliegues
├── .env.example                    # Plantilla de variables de entorno
├── Procfile / runtime.txt          # Despliegue de la app Streamlit en Heroku/Railway
└── .dockerignore
```

## 🚀 Instalación

```bash
git clone https://github.com/Jhonatan0828/fraudes_en_seguros_veh.git
cd fraudes_en_seguros_veh

uv venv                     # o: python -m venv .venv
.venv\Scripts\activate      # Windows   (Linux/macOS: source .venv/bin/activate)
uv pip install -e ".[dev]"  # o: pip install -e ".[dev]"

copy .env.example .env      # Windows   (Linux/macOS: cp .env.example .env)
```

El dataset original viene incluido en `data/raw/fraud_oracle.csv`, así que el
pipeline se puede ejecutar recién clonado el repositorio.

> **Windows — límite de rutas de 260 caracteres.** Prefect instala archivos con
> rutas de hasta 339 caracteres (los bundles de su interfaz web). Si el
> repositorio está en una carpeta profunda, el flow falla al arrancar con
> `Ephemeral server process exited with code 3`. Los tests y la API sí funcionan;
> solo Prefect se ve afectado.
>
> Solución (PowerShell **como administrador**, una sola vez, y reiniciar):
> ```powershell
> Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled 1
> ```
> Sin permisos de administrador: clona el repositorio en una ruta corta
> (`C:\proyectos\fraude`) o crea el entorno fuera de la carpeta profunda
> (`uv venv C:\venvs\fraude`).

## ▶️ Ejecutar el pipeline

```bash
python -m flows.training_flow                 # pipeline completo
python -m flows.training_flow --no-tuning     # sin búsqueda de hiperparámetros
python -m flows.training_flow --source url --url https://…
python -m flows.deployment                    # deja el flow programado (lunes 3:00)
```

Interfaces:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
prefect server start                               # http://localhost:4200
```

## 📊 Resultados

Una ejecución del flow produce 17 runs en MLflow: 1 de búsqueda de
hiperparámetros, 12 candidatos anidados y 4 modelos entrenados.

### Calidad de los datos

15.420 filas · 33 columnas · 0 duplicados · 0 nulos explícitos.
La etapa de validación detecta los **faltantes encubiertos** que el EDA
identificó: `Age = 0` (320 registros), `MonthClaimed = '0'` y
`DayOfWeekClaimed = '0'` (1 cada uno). Tasa de fraude: **5,99%**
(1 fraude por cada 15,7 reclamaciones legítimas).

### Comparación de candidatos

| Modelo | PR-AUC | ROC-AUC | Umbral | F1 | Recall | Precisión |
|:---|---:|---:|---:|---:|---:|---:|
| **XGBoost** ⭐ | **0.2413** | **0.8291** | 0.6589 | **0.2798** | 0.5892 | 0.1835 |
| Random Forest | 0.2024 | 0.8151 | 0.5995 | 0.2632 | 0.6054 | 0.1682 |
| Logistic Regression | 0.1696 | 0.8038 | 0.6756 | 0.2482 | 0.5514 | 0.1601 |
| Logistic + SMOTE | 0.1665 | 0.8016 | 0.6412 | 0.2475 | 0.6595 | 0.1523 |

F1, recall y precisión están medidos en el umbral óptimo de cada modelo, no en
0.5. Tabla completa en [`reports/model_comparison.md`](reports/model_comparison.md).

### Modelo candidato a producción

🏆 **XGBoost** — registrado como `fraud-detection-classifier` v1 con alias
`champion`, umbral de decisión **0.6589**.

**Por qué PR-AUC:** con 5,99% de clase positiva, el accuracy premia a un modelo
que nunca detecte fraude (94% acertando siempre "legítimo") y el ROC-AUC se
infla con los verdaderos negativos, que aquí sobran. PR-AUC mide solo el
desempeño sobre la clase que importa.

**Por qué no el umbral 0.5:** a 0.5 el modelo marca el 38% de las
reclamaciones (recall 0.92, precisión 0.13) — inviable para un equipo de
analistas. En 0.6589 el F1 sube de 0.225 a 0.280: detecta el 59% de los
fraudes revisando menos de la mitad de casos. El umbral se recalcula en cada
ejecución y viaja como tag de la versión del modelo.

**Compuerta de calidad:** el flow solo promueve si el mejor candidato supera
`MIN_PR_AUC` (0.20 por defecto); si no, falla y el campeón anterior sigue en
producción.

### Optimización de hiperparámetros

`RandomizedSearchCV` — 12 combinaciones × 3 folds estratificados, optimizando
PR-AUC. Mejor resultado en validación cruzada: **0.2325** con
`n_estimators=300, max_depth=4, learning_rate=0.01, subsample=0.7,
colsample_bytree=0.8, min_child_weight=1, gamma=1.0`.

## 🌐 Despliegue

### Servicio REST local

```bash
uvicorn src.api.main:app --reload
```

Documentación interactiva en `http://localhost:8000/docs`.

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Estado del servicio y versión cargada |
| `/model-info` | GET | Modelo, versión, métrica de selección y umbral |
| `/predict` | POST | Evalúa una reclamación |
| `/predict/batch` | POST | Evalúa un lote de reclamaciones |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Month":"Dec","DayOfWeek":"Wednesday","AccidentArea":"Urban",
       "Fault":"Policy Holder","MonthClaimed":"Jan","DayOfWeekClaimed":"Tuesday",
       "Days_Policy_Accident":"more than 30","Days_Policy_Claim":"more than 30",
       "PoliceReportFiled":"No","WitnessPresent":"No","NumberOfSuppliments":"none",
       "Age":21,"Sex":"Female","MaritalStatus":"Single","PastNumberOfClaims":"none",
       "AddressChange_Claim":"1 year","NumberOfCars":"3 to 4","Make":"Honda",
       "VehicleCategory":"Sport","VehiclePrice":"more than 69000",
       "AgeOfVehicle":"3 years","PolicyType":"Sport - Liability",
       "BasePolicy":"Liability","Deductible":300,"DriverRating":1,
       "AgentType":"External"}'
```

### Docker

```bash
docker compose -f deployment/docker-compose.yml up --build
```

La imagen empaqueta `models/champion/` y se instala con el `requirements.txt`
que MLflow genera junto al modelo, de modo que el contenedor reproduce las
versiones exactas con las que se entrenó.

Para servir directamente desde el Model Registry en lugar del export local:

```bash
MODEL_URI=models:/fraud-detection-classifier@champion uvicorn src.api.main:app
```

## 🔄 Promover un modelo nuevo

El servicio no conoce nombres de modelos: sigue el alias `champion`. Promover
una versión distinta no requiere tocar código ni reconstruir la imagen si se
sirve desde el registro.

```python
from mlflow.tracking import MlflowClient
MlflowClient().set_registered_model_alias("fraud-detection-classifier", "champion", version=3)
```

## 🖥️ App de exploración (Streamlit)

```bash
streamlit run app.py
```

EDA interactivo con 15 filtros, comparación de modelos y predicción individual.
Es la herramienta de exploración de la fase de análisis: consume los artefactos
de `models/*.pkl` que genera el notebook, mientras que la ruta de producción es
el modelo `champion` que sirve la API. El `Procfile` la despliega en
Heroku/Railway.

## 🧪 Tests

```bash
pytest tests/ -v
```

30 tests sobre las etapas del pipeline, los endpoints de la API y los
utilitarios de la app. Los de la API se omiten automáticamente si todavía no
existe `models/champion/`. Se ejecutan en cada `push` y `pull request`
(GitHub Actions).

## 👥 Equipo

- Andres Felipe Londoño Ocampo
- Jhonatan Caro Atehortúa
- Paulina Perez Ramirez
- Victor Manuel Galeano Alvarez

## 🛠️ Stack tecnológico

Python 3.11 · **Prefect** · **MLflow** · **FastAPI** · **Docker** ·
Scikit-Learn · XGBoost · imbalanced-learn · Pandas · Plotly · Streamlit · pytest

## 📄 Licencia

Proyecto académico — Universidad de Medellín, 2026.
